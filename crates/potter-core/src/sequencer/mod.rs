use serde::{Deserialize, Serialize};

use crate::{
    color::linear_to_srgb_f32,
    error::{ErrorCode, PotError, Result},
    media::AudioBuffer,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Sequencer {
    pub channels: u32,
    pub strips: Vec<Strip>,
}

impl Default for Sequencer {
    fn default() -> Self {
        Self {
            channels: 32,
            strips: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Strip {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub kind: StripType,
    pub channel: u32,
    pub frame_start: f64,
    pub frame_offset_start: f64,
    pub frame_offset_end: f64,
    pub length: f64,
    pub blend_type: StripBlendType,
    pub opacity: f64,
    pub mute: bool,
    pub retiming_keys: Vec<RetimingKey>,
    pub modifiers: Vec<StripModifier>,
    pub sound_volume: f64,
    pub sound_pan: f64,
    pub sound_pitch: f64,
    pub source: Option<String>,
    pub color: [f32; 4],
    pub text: Option<String>,
    pub transition: Option<TransitionType>,
    pub inputs: Vec<String>,
    pub effect: Option<EffectType>,
}

impl Default for Strip {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            kind: StripType::Image,
            channel: 1,
            frame_start: 1.0,
            frame_offset_start: 0.0,
            frame_offset_end: 0.0,
            length: 1.0,
            blend_type: StripBlendType::Replace,
            opacity: 1.0,
            mute: false,
            retiming_keys: Vec::new(),
            modifiers: Vec::new(),
            sound_volume: 1.0,
            sound_pan: 0.0,
            sound_pitch: 1.0,
            source: None,
            color: [0.0, 0.0, 0.0, 1.0],
            text: None,
            transition: None,
            inputs: Vec::new(),
            effect: None,
        }
    }
}

impl Strip {
    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty()
            || self.channel == 0
            || !self.frame_start.is_finite()
            || !self.frame_offset_start.is_finite()
            || !self.frame_offset_end.is_finite()
            || self.frame_offset_start < 0.0
            || self.frame_offset_end < 0.0
            || !self.length.is_finite()
            || self.length <= 0.0
            || self.frame_offset_start + self.frame_offset_end >= self.length
            || !self.opacity.is_finite()
            || !(0.0..=1.0).contains(&self.opacity)
            || !self.sound_volume.is_finite()
            || self.sound_volume < 0.0
            || !self.sound_pan.is_finite()
            || !(-1.0..=1.0).contains(&self.sound_pan)
            || !self.sound_pitch.is_finite()
            || self.sound_pitch <= 0.0
            || self.color.iter().any(|component| !component.is_finite())
            || (self.kind == StripType::Transition) != self.transition.is_some()
            || (self.kind == StripType::Effect) != self.effect.is_some()
            || self.modifiers.iter().any(|modifier| !modifier.is_valid())
        {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "sequencer strip has invalid identity, timing, channel, or settings",
            ));
        }
        let expected_inputs = match (self.kind, self.effect) {
            (StripType::Transition, _)
            | (
                StripType::Effect,
                Some(
                    EffectType::Add
                    | EffectType::Subtract
                    | EffectType::Multiply
                    | EffectType::AlphaOver,
                ),
            ) => Some(2),
            (StripType::Effect, Some(_)) => Some(1),
            _ => None,
        };
        if expected_inputs.is_some_and(|count| count != self.inputs.len())
            || self.inputs.iter().enumerate().any(|(index, input)| {
                input.is_empty() || input == &self.id || self.inputs[..index].contains(input)
            })
        {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "sequencer transition or effect inputs are invalid",
            ));
        }
        let end = self.frame_start + self.length;
        if !end.is_finite() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "sequencer strip end frame exceeds finite range",
            ));
        }
        let mut previous = f64::NEG_INFINITY;
        for key in &self.retiming_keys {
            if !key.frame.is_finite() || !key.source_frame.is_finite() || key.frame <= previous {
                return Err(PotError::new(
                    ErrorCode::InvalidOperation,
                    "sequencer retiming keys must be finite and strictly ordered",
                ));
            }
            previous = key.frame;
        }
        Ok(())
    }

    #[must_use]
    pub fn contains_frame(&self, frame: f64) -> bool {
        frame.is_finite() && frame >= self.frame_start && frame < self.frame_start + self.length
    }

    #[must_use]
    pub fn source_frame(&self, frame: f64) -> f64 {
        let timeline_offset = frame - self.frame_start;
        let duration = (self.length - self.frame_offset_start - self.frame_offset_end).max(0.0);
        if duration == 0.0 {
            return self.frame_offset_start;
        }
        if let Some(retimed) = retimed_source_frame(&self.retiming_keys, frame) {
            return retimed;
        }
        let normalized = (timeline_offset / self.length).clamp(0.0, 1.0);
        self.frame_offset_start + normalized * duration
    }
}

fn retimed_source_frame(keys: &[RetimingKey], frame: f64) -> Option<f64> {
    if keys.is_empty() {
        return None;
    }
    let upper = keys.partition_point(|key| key.frame <= frame);
    Some(match upper {
        0 => keys[0].source_frame,
        index if index >= keys.len() => keys[keys.len() - 1].source_frame,
        index => {
            let left = &keys[index - 1];
            let right = &keys[index];
            let factor = (frame - left.frame) / (right.frame - left.frame);
            left.source_frame + (right.source_frame - left.source_frame) * factor
        }
    })
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum StripType {
    #[default]
    Image,
    ImageSequence,
    Movie,
    Sound,
    Scene,
    Color,
    Text,
    Meta,
    Transition,
    Effect,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum StripBlendType {
    #[default]
    Replace,
    AlphaOver,
    Add,
    Subtract,
    Multiply,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TransitionType {
    Cross,
    GammaCross,
    Wipe,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EffectType {
    Add,
    Subtract,
    Multiply,
    AlphaOver,
    Transform,
    Speed,
    Glow,
    GaussianBlur,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformSettings {
    pub translation: [f64; 2],
    pub scale: [f64; 2],
    pub rotation_radians: f64,
}

impl Default for TransformSettings {
    fn default() -> Self {
        Self {
            translation: [0.0; 2],
            scale: [1.0; 2],
            rotation_radians: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlowSettings {
    pub threshold: f32,
    pub intensity: f32,
    pub radius: u32,
}

impl Default for GlowSettings {
    fn default() -> Self {
        Self {
            threshold: 0.8,
            intensity: 1.0,
            radius: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RetimingKey {
    pub frame: f64,
    pub source_frame: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct StripModifier {
    pub brightness: f64,
    pub contrast: f64,
    pub color_balance: [f64; 3],
    pub curves: Vec<[f64; 2]>,
}

impl Default for StripModifier {
    fn default() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.0,
            color_balance: [1.0; 3],
            curves: Vec::new(),
        }
    }
}

impl StripModifier {
    fn is_valid(&self) -> bool {
        self.brightness.is_finite()
            && self.contrast.is_finite()
            && self.color_balance.iter().all(|value| value.is_finite())
            && self
                .curves
                .iter()
                .all(|point| point[0].is_finite() && point[1].is_finite())
            && self
                .curves
                .windows(2)
                .all(|points| points[0][0] < points[1][0])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RgbaFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 4]>,
}

impl RgbaFrame {
    pub fn new(width: u32, height: u32, pixels: Vec<[f32; 4]>) -> Result<Self> {
        let expected = usize::try_from(width)
            .ok()
            .and_then(|width| {
                usize::try_from(height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| {
                PotError::new(ErrorCode::InvalidArgument, "image dimensions overflow")
            })?;
        if width == 0 || height == 0 || pixels.len() != expected {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "RGBA frame dimensions do not match pixel data",
            ));
        }
        if pixels.iter().flatten().any(|value| !value.is_finite()) {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "RGBA frame contains non-finite pixels",
            ));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }
}

pub fn crossfade(first: &RgbaFrame, second: &RgbaFrame, factor: f64) -> Result<RgbaFrame> {
    validate_frames(first, second)?;
    if !factor.is_finite() || !(0.0..=1.0).contains(&factor) {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "crossfade factor must be in [0, 1]",
        ));
    }
    let amount = factor as f32;
    let pixels = first
        .pixels
        .iter()
        .zip(&second.pixels)
        .map(|(a, b)| {
            [
                lerp(a[0], b[0], amount),
                lerp(a[1], b[1], amount),
                lerp(a[2], b[2], amount),
                lerp(a[3], b[3], amount),
            ]
        })
        .collect();
    RgbaFrame::new(first.width, first.height, pixels)
}

pub fn blend_frames(
    first: &RgbaFrame,
    second: &RgbaFrame,
    blend_type: StripBlendType,
    opacity: f64,
) -> Result<RgbaFrame> {
    validate_frames(first, second)?;
    if !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "blend opacity must be in [0, 1]",
        ));
    }
    let opacity = opacity as f32;
    let pixels = first
        .pixels
        .iter()
        .zip(&second.pixels)
        .map(|(destination, source)| {
            let source_alpha = source[3] * opacity;
            let mut result = [0.0; 4];
            if blend_type == StripBlendType::AlphaOver {
                let output_alpha = source_alpha + destination[3] * (1.0 - source_alpha);
                result[3] = output_alpha;
                if output_alpha > 0.0 {
                    for channel in 0..3 {
                        result[channel] = (source[channel] * source_alpha
                            + destination[channel] * destination[3] * (1.0 - source_alpha))
                            / output_alpha;
                    }
                }
            } else if blend_type == StripBlendType::Replace {
                for channel in 0..4 {
                    result[channel] = lerp(destination[channel], source[channel], opacity);
                }
            } else {
                for channel in 0..3 {
                    let blended = match blend_type {
                        StripBlendType::Subtract => destination[channel] - source[channel],
                        StripBlendType::Multiply => destination[channel] * source[channel],
                        StripBlendType::Add => destination[channel] + source[channel],
                        StripBlendType::Replace | StripBlendType::AlphaOver => destination[channel],
                    };
                    result[channel] = lerp(destination[channel], blended, source_alpha);
                }
                result[3] = source_alpha + destination[3] * (1.0 - source_alpha);
            }
            result
        })
        .collect();
    RgbaFrame::new(first.width, first.height, pixels)
}

pub fn apply_blend_effect(
    effect: EffectType,
    first: &RgbaFrame,
    second: &RgbaFrame,
    factor: f64,
) -> Result<RgbaFrame> {
    let blend_type = match effect {
        EffectType::Add => StripBlendType::Add,
        EffectType::Subtract => StripBlendType::Subtract,
        EffectType::Multiply => StripBlendType::Multiply,
        EffectType::AlphaOver => StripBlendType::AlphaOver,
        EffectType::Transform | EffectType::Speed | EffectType::Glow | EffectType::GaussianBlur => {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "single-input effects must use apply_effect",
            ));
        }
    };
    blend_frames(first, second, blend_type, factor)
}

pub fn apply_effect(effect: EffectType, frame: RgbaFrame) -> Result<RgbaFrame> {
    match effect {
        EffectType::Transform => transform_frame(frame, TransformSettings::default()),
        EffectType::Speed => Ok(frame),
        EffectType::Glow => apply_glow(&frame, GlowSettings::default()),
        EffectType::GaussianBlur => gaussian_blur(frame, 1),
        EffectType::Add | EffectType::Subtract | EffectType::Multiply | EffectType::AlphaOver => {
            Err(PotError::new(
                ErrorCode::InvalidArgument,
                "two-input effects must use apply_blend_effect",
            ))
        }
    }
}

pub fn transform_frame(frame: RgbaFrame, settings: TransformSettings) -> Result<RgbaFrame> {
    validate_frames(&frame, &frame)?;
    if settings.translation.iter().any(|value| !value.is_finite())
        || settings
            .scale
            .iter()
            .any(|value| !value.is_finite() || *value == 0.0)
        || !settings.rotation_radians.is_finite()
    {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "transform effect settings are invalid",
        ));
    }
    if settings == TransformSettings::default() {
        return Ok(frame);
    }
    let width = usize::try_from(frame.width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "frame width exceeds platform range",
        )
    })?;
    let center_x = (f64::from(frame.width) - 1.0) * 0.5;
    let center_y = (f64::from(frame.height) - 1.0) * 0.5;
    let (sin, cos) = settings.rotation_radians.sin_cos();
    let mut pixels = Vec::with_capacity(frame.pixels.len());
    for y in 0..frame.height {
        for x in 0..frame.width {
            let dx = f64::from(x) - center_x - settings.translation[0];
            let dy = f64::from(y) - center_y - settings.translation[1];
            let source_x = center_x + (cos * dx + sin * dy) / settings.scale[0];
            let source_y = center_y + (-sin * dx + cos * dy) / settings.scale[1];
            pixels.push(bilinear_sample(&frame, width, source_x, source_y)?);
        }
    }
    RgbaFrame::new(frame.width, frame.height, pixels)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "sample coordinates are finite and checked against frame bounds"
)]
fn bilinear_sample(frame: &RgbaFrame, width: usize, x: f64, y: f64) -> Result<[f32; 4]> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x > f64::from(frame.width - 1)
        || y > f64::from(frame.height - 1)
    {
        return Ok([0.0; 4]);
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = x0.saturating_add(1).min(width - 1);
    let height = usize::try_from(frame.height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "frame height exceeds platform range",
        )
    })?;
    let y1 = y0.saturating_add(1).min(height - 1);
    let factor_x = (x - x0 as f64) as f32;
    let factor_y = (y - y0 as f64) as f32;
    let upper_left = frame.pixels[y0 * width + x0];
    let upper_right = frame.pixels[y0 * width + x1];
    let lower_left = frame.pixels[y1 * width + x0];
    let lower_right = frame.pixels[y1 * width + x1];
    let mut pixel = [0.0; 4];
    for channel in 0..4 {
        let upper = lerp(upper_left[channel], upper_right[channel], factor_x);
        let lower = lerp(lower_left[channel], lower_right[channel], factor_x);
        pixel[channel] = lerp(upper, lower, factor_y);
    }
    Ok(pixel)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "bounded blur kernel and channel values are converted to computation types"
)]
pub fn gaussian_blur(frame: RgbaFrame, radius: u32) -> Result<RgbaFrame> {
    validate_frames(&frame, &frame)?;
    if radius > 128 {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "Gaussian blur radius exceeds 128 pixels",
        ));
    }
    if radius == 0 {
        return Ok(frame);
    }
    let width = usize::try_from(frame.width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "frame width exceeds platform range",
        )
    })?;
    let height = usize::try_from(frame.height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "frame height exceeds platform range",
        )
    })?;
    let radius = usize::try_from(radius).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "Gaussian blur radius exceeds platform range",
        )
    })?;
    let kernel_length = radius
        .checked_mul(2)
        .and_then(|length| length.checked_add(1))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "Gaussian blur kernel is too large",
            )
        })?;
    let sigma = (radius as f64 * 0.5).max(0.5);
    let mut kernel = Vec::with_capacity(kernel_length);
    let mut kernel_sum = 0.0;
    for index in 0..kernel_length {
        let offset = i64::try_from(index).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "Gaussian blur kernel is too large",
            )
        })? - i64::try_from(radius).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "Gaussian blur kernel is too large",
            )
        })?;
        let offset = offset as f64;
        let weight = (-(offset * offset) / (2.0 * sigma * sigma)).exp();
        kernel.push(weight);
        kernel_sum += weight;
    }
    for weight in &mut kernel {
        *weight /= kernel_sum;
    }
    let mut horizontal = vec![[0.0_f64; 4]; frame.pixels.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = [0.0; 4];
            for (kernel_index, weight) in kernel.iter().enumerate() {
                let sample_x = x
                    .saturating_add(kernel_index)
                    .saturating_sub(radius)
                    .min(width - 1);
                let source = frame.pixels[y * width + sample_x];
                let alpha = f64::from(source[3]);
                for channel in 0..3 {
                    sum[channel] += f64::from(source[channel]) * alpha * weight;
                }
                sum[3] += alpha * weight;
            }
            horizontal[y * width + x] = sum;
        }
    }
    let mut pixels = vec![[0.0_f32; 4]; frame.pixels.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = [0.0; 4];
            for (kernel_index, weight) in kernel.iter().enumerate() {
                let sample_y = y
                    .saturating_add(kernel_index)
                    .saturating_sub(radius)
                    .min(height - 1);
                let source = horizontal[sample_y * width + x];
                for channel in 0..4 {
                    sum[channel] += source[channel] * weight;
                }
            }
            let destination = &mut pixels[y * width + x];
            destination[3] = sum[3] as f32;
            if sum[3] > 0.0 {
                for channel in 0..3 {
                    destination[channel] = (sum[channel] / sum[3]) as f32;
                }
            }
        }
    }
    RgbaFrame::new(frame.width, frame.height, pixels)
}

pub fn apply_glow(frame: &RgbaFrame, settings: GlowSettings) -> Result<RgbaFrame> {
    validate_frames(frame, frame)?;
    if !settings.threshold.is_finite()
        || settings.threshold < 0.0
        || !settings.intensity.is_finite()
        || settings.intensity < 0.0
        || settings.radius > 128
    {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "glow effect settings are invalid",
        ));
    }
    let bright_pixels = frame
        .pixels
        .iter()
        .map(|pixel| {
            let brightness = pixel[0].max(pixel[1]).max(pixel[2]);
            let factor = if brightness > settings.threshold && brightness > 0.0 {
                (brightness - settings.threshold) / brightness
            } else {
                0.0
            };
            [
                pixel[0] * factor,
                pixel[1] * factor,
                pixel[2] * factor,
                pixel[3] * factor,
            ]
        })
        .collect();
    let bright = RgbaFrame::new(frame.width, frame.height, bright_pixels)?;
    let glow = gaussian_blur(bright, settings.radius)?;
    let pixels = frame
        .pixels
        .iter()
        .zip(glow.pixels)
        .map(|(base, glow)| {
            [
                base[0] + glow[0] * settings.intensity,
                base[1] + glow[1] * settings.intensity,
                base[2] + glow[2] * settings.intensity,
                base[3],
            ]
        })
        .collect();
    RgbaFrame::new(frame.width, frame.height, pixels)
}

pub fn apply_modifiers(frame: &RgbaFrame, modifiers: &[StripModifier]) -> Result<RgbaFrame> {
    validate_frames(frame, frame)?;
    let mut pixels = frame.pixels.clone();
    for modifier in modifiers {
        if !modifier.is_valid() {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "sequencer modifier settings are invalid",
            ));
        }
        for pixel in &mut pixels {
            for (channel, balance) in modifier.color_balance.iter().enumerate() {
                let adjusted = ((f64::from(pixel[channel]) - 0.5) * modifier.contrast
                    + 0.5
                    + modifier.brightness)
                    * balance;
                pixel[channel] = sample_modifier_curve(adjusted, &modifier.curves) as f32;
            }
        }
    }
    RgbaFrame::new(frame.width, frame.height, pixels)
}

fn sample_modifier_curve(value: f64, curve: &[[f64; 2]]) -> f64 {
    let Some(first) = curve.first() else {
        return value;
    };
    if value <= first[0] {
        return first[1];
    }
    let upper = curve.partition_point(|point| point[0] <= value);
    if upper == curve.len() {
        return curve[curve.len() - 1][1];
    }
    let left = curve[upper - 1];
    let right = curve[upper];
    let factor = (value - left[0]) / (right[0] - left[0]);
    left[1] + (right[1] - left[1]) * factor
}

pub fn transition(
    first: &RgbaFrame,
    second: &RgbaFrame,
    factor: f64,
    kind: TransitionType,
) -> Result<RgbaFrame> {
    validate_frames(first, second)?;
    if !factor.is_finite() || !(0.0..=1.0).contains(&factor) {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "transition factor must be in [0, 1]",
        ));
    }
    match kind {
        TransitionType::Cross => crossfade(first, second, factor),
        TransitionType::GammaCross => gamma_cross(first, second, factor),
        TransitionType::Wipe => wipe(first, second, factor),
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "transition factor is validated in [0, 1] before channel conversion"
)]
fn gamma_cross(first: &RgbaFrame, second: &RgbaFrame, factor: f64) -> Result<RgbaFrame> {
    let amount = factor as f32;
    let pixels = first
        .pixels
        .iter()
        .zip(&second.pixels)
        .map(|(a, b)| {
            let mut pixel = [0.0; 4];
            for channel in 0..3 {
                let first_linear = srgb_to_linear(a[channel]);
                let second_linear = srgb_to_linear(b[channel]);
                pixel[channel] = linear_to_srgb_f32(lerp(first_linear, second_linear, amount));
            }
            pixel[3] = lerp(a[3], b[3], amount);
            pixel
        })
        .collect();
    RgbaFrame::new(first.width, first.height, pixels)
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "factor is validated in [0, 1] and the rounded wipe cut stays within frame width"
)]
fn wipe(first: &RgbaFrame, second: &RgbaFrame, factor: f64) -> Result<RgbaFrame> {
    let cut = (f64::from(first.width) * factor).round() as u32;
    let width = usize::try_from(first.width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "frame width exceeds platform range",
        )
    })?;
    let height = usize::try_from(first.height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "frame height exceeds platform range",
        )
    })?;
    let cut = usize::try_from(cut).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "wipe position exceeds platform range",
        )
    })?;
    let mut pixels = first.pixels.clone();
    for y in 0..height {
        for x in 0..cut {
            let index = y * width + x;
            pixels[index] = second.pixels[index];
        }
    }
    RgbaFrame::new(first.width, first.height, pixels)
}

fn validate_frames(first: &RgbaFrame, second: &RgbaFrame) -> Result<()> {
    if first.width == 0
        || first.height == 0
        || first.width != second.width
        || first.height != second.height
    {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "frames must have matching nonzero dimensions",
        ));
    }
    let expected = usize::try_from(first.width)
        .ok()
        .and_then(|width| {
            usize::try_from(first.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| PotError::new(ErrorCode::InvalidArgument, "image dimensions overflow"))?;
    if first.pixels.len() != expected || second.pixels.len() != expected {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "transition frame pixel count does not match dimensions",
        ));
    }
    if first
        .pixels
        .iter()
        .chain(&second.pixels)
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "transition frame contains non-finite pixels",
        ));
    }
    Ok(())
}

fn lerp(first: f32, second: f32, factor: f32) -> f32 {
    first + (second - first) * factor
}

#[derive(Debug, Clone, Copy)]
pub struct AudioTrack<'a> {
    pub audio: &'a AudioBuffer,
    pub frame_start: f64,
    pub frame_end: Option<f64>,
    pub frame_offset_start: f64,
    pub frame_offset_end: f64,
    pub retiming_keys: &'a [RetimingKey],
    pub volume: f64,
    pub pan: f64,
    pub pitch: f64,
}

impl<'a> AudioTrack<'a> {
    #[must_use]
    pub const fn new(audio: &'a AudioBuffer, frame_start: f64) -> Self {
        Self {
            audio,
            frame_start,
            frame_end: None,
            frame_offset_start: 0.0,
            frame_offset_end: 0.0,
            retiming_keys: &[],
            volume: 1.0,
            pan: 0.0,
            pitch: 1.0,
        }
    }
}

pub fn frame_to_sample(frame: f64, fps: f64, sample_rate: u32) -> Result<i64> {
    if !frame.is_finite() || !fps.is_finite() || fps <= 0.0 || sample_rate == 0 {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "frame must be finite and fps and sample rate must be positive",
        ));
    }
    let sample = (frame / fps) * f64::from(sample_rate);
    let rounded = sample.round();
    let positive_limit = -(i64::MIN as f64);
    if !rounded.is_finite() || rounded < i64::MIN as f64 || rounded >= positive_limit {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "frame sample offset exceeds supported range",
        ));
    }
    Ok(rounded as i64)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "sample indexing conversions follow validated frame, rate, and buffer bounds"
)]
pub fn mix_audio(
    tracks: &[AudioTrack<'_>],
    frame_start: f64,
    frame_end: f64,
    fps: f64,
    output_rate: u32,
) -> Result<AudioBuffer> {
    if !frame_start.is_finite()
        || !frame_end.is_finite()
        || frame_end < frame_start
        || output_rate == 0
    {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "audio render range or sample rate is invalid",
        ));
    }
    for track in tracks {
        validate_audio_track(track)?;
    }
    let start_sample = frame_to_sample(frame_start, fps, output_rate)?;
    let end_sample = frame_to_sample(frame_end, fps, output_rate)?;
    let sample_span = end_sample.checked_sub(start_sample).ok_or_else(|| {
        PotError::new(ErrorCode::LimitExceeded, "audio render range is too large")
    })?;
    let sample_count = usize::try_from(sample_span)
        .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "audio render range is too large"))?;
    let channels = tracks
        .iter()
        .map(|track| track.audio.channels)
        .max()
        .unwrap_or(2)
        .max(if tracks.iter().any(|track| track.pan != 0.0) {
            2
        } else {
            1
        });
    if channels == 0 || channels > 8 {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "audio channel count must be between 1 and 8",
        ));
    }
    let sample_capacity = sample_count
        .checked_mul(usize::from(channels))
        .ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "audio render buffer is too large")
        })?;
    let mut samples = vec![0.0_f32; sample_capacity];
    for track in tracks {
        let track_start = frame_to_sample(track.frame_start, fps, output_rate)?;
        let track_end = track
            .frame_end
            .map(|frame| frame_to_sample(frame, fps, output_rate))
            .transpose()?;
        let source_offset = if track.retiming_keys.is_empty() {
            frame_to_sample(track.frame_offset_start, fps, track.audio.sample_rate)? as f64
        } else {
            0.0
        };
        let retimed_start_frame = retimed_source_frame(track.retiming_keys, track.frame_start);
        let retimed_start_sample = retimed_start_frame
            .map(|source_frame| (source_frame / fps) * f64::from(track.audio.sample_rate));
        if retimed_start_sample.is_some_and(|source_sample| !source_sample.is_finite()) {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "audio retiming start maps outside the source range",
            ));
        }
        let source_end_trim =
            frame_to_sample(track.frame_offset_end, fps, track.audio.sample_rate)?;
        let source_end_trim = usize::try_from(source_end_trim).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "audio source trim exceeds supported range",
            )
        })?;
        let source_channels = usize::from(track.audio.channels);
        let target_channels = usize::from(channels);
        let source_frame_count = track.audio.samples.len() / source_channels;
        let source_frame_end = source_frame_count.saturating_sub(source_end_trim);
        for output_index in 0..sample_count {
            let sample_offset = i64::try_from(output_index).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "audio sample index exceeds supported range",
                )
            })?;
            let timeline_sample = start_sample.checked_add(sample_offset).ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "audio sample index exceeds supported range",
                )
            })?;
            if timeline_sample < track_start || track_end.is_some_and(|end| timeline_sample >= end)
            {
                continue;
            }
            let elapsed = timeline_sample.checked_sub(track_start).ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "audio sample offset exceeds supported range",
                )
            })?;
            let source_position = match (retimed_start_frame, retimed_start_sample) {
                (Some(_), Some(start_sample)) => {
                    let timeline_frame = (timeline_sample as f64 / f64::from(output_rate)) * fps;
                    let source_frame = retimed_source_frame(track.retiming_keys, timeline_frame)
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "audio retiming keys disappeared",
                            )
                        })?;
                    let mapped_sample = (source_frame / fps) * f64::from(track.audio.sample_rate);
                    start_sample + (mapped_sample - start_sample) * track.pitch
                }
                (None, None) => {
                    source_offset
                        + (elapsed as f64) * f64::from(track.audio.sample_rate)
                            / f64::from(output_rate)
                            * track.pitch
                }
                _ => {
                    return Err(PotError::new(
                        ErrorCode::InternalError,
                        "audio retiming state is inconsistent",
                    ));
                }
            };
            if !source_position.is_finite() {
                return Err(PotError::new(
                    ErrorCode::InvalidArgument,
                    "audio source mapping exceeds finite range",
                ));
            }
            if source_position < 0.0 || source_position >= source_frame_end as f64 {
                continue;
            }
            let source_floor = source_position.floor() as usize;
            let source_next = (source_floor + 1).min(source_frame_end.saturating_sub(1));
            let source_fraction = (source_position - source_floor as f64) as f32;
            for channel in 0..target_channels {
                let source_channel = if source_channels == 1 {
                    0
                } else {
                    channel.min(source_channels - 1)
                };
                let first = track.audio.samples[source_floor * source_channels + source_channel];
                let second = track.audio.samples[source_next * source_channels + source_channel];
                let mut gain = track.volume;
                if target_channels >= 2 {
                    if channel == 0 && track.pan > 0.0 {
                        gain *= 1.0 - track.pan;
                    } else if channel == 1 && track.pan < 0.0 {
                        gain *= 1.0 + track.pan;
                    }
                }
                let destination = output_index * target_channels + channel;
                samples[destination] += lerp(first, second, source_fraction) * gain as f32;
            }
        }
    }
    for sample in &mut samples {
        *sample = sample.clamp(-1.0, 1.0);
    }
    Ok(AudioBuffer {
        sample_rate: output_rate,
        channels,
        samples,
    })
}

fn validate_audio_track(track: &AudioTrack<'_>) -> Result<()> {
    if track.audio.channels == 0
        || track.audio.sample_rate == 0
        || !track
            .audio
            .samples
            .len()
            .is_multiple_of(usize::from(track.audio.channels))
        || !track.frame_start.is_finite()
        || track
            .frame_end
            .is_some_and(|end| !end.is_finite() || end < track.frame_start)
        || !track.frame_offset_start.is_finite()
        || track.frame_offset_start < 0.0
        || !track.frame_offset_end.is_finite()
        || track.frame_offset_end < 0.0
        || !track.volume.is_finite()
        || track.volume < 0.0
        || !track.pan.is_finite()
        || !(-1.0..=1.0).contains(&track.pan)
        || !track.pitch.is_finite()
        || track.pitch <= 0.0
        || track
            .retiming_keys
            .iter()
            .any(|key| !key.frame.is_finite() || !key.source_frame.is_finite())
        || track
            .retiming_keys
            .windows(2)
            .any(|keys| keys[0].frame >= keys[1].frame)
    {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "audio strip contains invalid samples or settings",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "sequencer regression tests")]

    use crate::media::AudioBuffer;

    use super::{
        AudioTrack, EffectType, GlowSettings, RgbaFrame, StripBlendType, StripModifier,
        TransformSettings, TransitionType, apply_blend_effect, apply_effect, apply_glow,
        apply_modifiers, blend_frames, crossfade, frame_to_sample, mix_audio, transform_frame,
        transition,
    };

    fn assert_pixel_close(actual: [f32; 4], expected: [f32; 4]) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn crossfade_at_midpoint_averages_rgba() {
        let left = RgbaFrame {
            width: 1,
            height: 1,
            pixels: vec![[0.0, 0.2, 0.4, 1.0]],
        };
        let right = RgbaFrame {
            width: 1,
            height: 1,
            pixels: vec![[1.0, 0.8, 0.6, 1.0]],
        };
        let mixed = crossfade(&left, &right, 0.5).unwrap();
        assert_eq!(mixed.pixels[0], [0.5, 0.5, 0.5, 1.0]);
    }

    #[test]
    fn gamma_cross_blends_in_linear_light() {
        let black = RgbaFrame::new(1, 1, vec![[0.0, 0.0, 0.0, 1.0]]).unwrap();
        let white = RgbaFrame::new(1, 1, vec![[1.0, 1.0, 1.0, 1.0]]).unwrap();
        let mixed = transition(&black, &white, 0.5, TransitionType::GammaCross).unwrap();
        assert!((mixed.pixels[0][0] - 0.735_356_9).abs() < 1.0e-5);
    }

    #[test]
    fn wipe_transition_replaces_the_entered_horizontal_region() {
        let first = RgbaFrame::new(2, 1, vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]).unwrap();
        let second =
            RgbaFrame::new(2, 1, vec![[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]).unwrap();
        let mixed = transition(&first, &second, 0.5, TransitionType::Wipe).unwrap();
        assert_eq!(
            mixed.pixels,
            vec![[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
        );
    }

    #[test]
    fn alpha_over_composites_using_source_alpha_and_strip_opacity() {
        let base = RgbaFrame::new(1, 1, vec![[0.2, 0.4, 0.6, 1.0]]).unwrap();
        let overlay = RgbaFrame::new(1, 1, vec![[1.0, 0.0, 0.0, 0.5]]).unwrap();
        let mixed = blend_frames(&base, &overlay, StripBlendType::AlphaOver, 0.5).unwrap();
        assert_pixel_close(mixed.pixels[0], [0.4, 0.3, 0.45, 1.0]);
    }

    #[test]
    fn strip_modifiers_apply_brightness_balance_and_curve() {
        let frame = RgbaFrame::new(1, 1, vec![[0.25, 0.5, 0.75, 0.4]]).unwrap();
        let modifier = StripModifier {
            brightness: 0.25,
            color_balance: [2.0, 1.0, 1.0],
            curves: vec![[0.0, 0.0], [1.0, 0.5]],
            ..StripModifier::default()
        };
        let adjusted = apply_modifiers(&frame, &[modifier]).unwrap();
        assert!((adjusted.pixels[0][0] - 0.5).abs() < 1.0e-6);
        assert!((adjusted.pixels[0][1] - 0.375).abs() < 1.0e-6);
        assert!((adjusted.pixels[0][2] - 0.5).abs() < 1.0e-6);
        assert_eq!(adjusted.pixels[0][3], 0.4);
    }

    #[test]
    fn parameterized_pixel_effects_have_deterministic_defaults() {
        let frame = RgbaFrame::new(2, 1, vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]).unwrap();
        let transformed = apply_effect(EffectType::Transform, frame.clone()).unwrap();
        assert_eq!(transformed, frame);
        let shifted = transform_frame(
            frame,
            TransformSettings {
                translation: [1.0, 0.0],
                ..TransformSettings::default()
            },
        )
        .unwrap();
        assert_eq!(shifted.pixels, vec![[0.0; 4], [1.0, 0.0, 0.0, 1.0]]);

        let impulse = RgbaFrame::new(
            3,
            1,
            vec![
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        .unwrap();
        let blurred = apply_effect(EffectType::GaussianBlur, impulse.clone()).unwrap();
        assert!(blurred.pixels[0][0] > 0.0);
        assert!(blurred.pixels[1][0] < 1.0);
        let glow = apply_glow(&impulse, GlowSettings::default()).unwrap();
        assert!(glow.pixels[0][0] > 0.0);
        assert_eq!(
            apply_effect(EffectType::Speed, impulse.clone()).unwrap(),
            impulse
        );
    }

    #[test]
    fn blend_effects_evaluate_add_subtract_multiply_and_alpha_over() {
        let first = RgbaFrame::new(1, 1, vec![[0.4, 0.6, 0.8, 1.0]]).unwrap();
        let second = RgbaFrame::new(1, 1, vec![[0.2, 0.3, 0.5, 1.0]]).unwrap();
        let added = apply_blend_effect(EffectType::Add, &first, &second, 0.5).unwrap();
        assert_pixel_close(added.pixels[0], [0.5, 0.75, 1.05, 1.0]);
        let subtracted = apply_blend_effect(EffectType::Subtract, &first, &second, 0.5).unwrap();
        assert_pixel_close(subtracted.pixels[0], [0.3, 0.45, 0.55, 1.0]);
        let multiplied = apply_blend_effect(EffectType::Multiply, &first, &second, 0.5).unwrap();
        assert_pixel_close(multiplied.pixels[0], [0.24, 0.39, 0.6, 1.0]);
    }

    #[test]
    fn audio_mix_places_sources_at_frame_accurate_sample_offsets() {
        let first = AudioBuffer {
            sample_rate: 48_000,
            channels: 1,
            samples: vec![0.25; 4_000],
        };
        let second = AudioBuffer {
            sample_rate: 48_000,
            channels: 1,
            samples: vec![0.5; 500],
        };
        let tracks = [AudioTrack::new(&first, 0.0), AudioTrack::new(&second, 1.0)];
        let mixed = mix_audio(&tracks, 0.0, 2.0, 24.0, 48_000).unwrap();
        assert_eq!(mixed.channels, 1);
        assert_eq!(mixed.samples.len(), 4_000);
        assert!(
            mixed.samples[..2_000]
                .iter()
                .all(|sample| (*sample - 0.25).abs() < 1.0e-6)
        );
        assert!(
            mixed.samples[2_000..2_500]
                .iter()
                .all(|sample| (*sample - 0.75).abs() < 1.0e-6)
        );
        assert!(
            mixed.samples[2_500..]
                .iter()
                .all(|sample| (*sample - 0.25).abs() < 1.0e-6)
        );
    }

    #[test]
    fn frame_to_sample_rounds_subframes_consistently() {
        assert_eq!(frame_to_sample(1.0, 24.0, 48_000).unwrap(), 2_000);
        assert_eq!(frame_to_sample(0.5, 24.0, 48_000).unwrap(), 1_000);
    }

    #[test]
    fn frame_to_sample_rejects_the_unrepresentable_positive_boundary() {
        let error = frame_to_sample(2.0_f64.powi(63), 1.0, 1).unwrap_err();
        assert_eq!(error.code, crate::error::ErrorCode::LimitExceeded);
    }

    #[test]
    fn audio_mix_rejects_ranges_whose_sample_span_overflows() {
        let error = mix_audio(
            &[],
            -100_000_000_000_000.0,
            100_000_000_000_000.0,
            1.0,
            48_000,
        )
        .unwrap_err();
        assert_eq!(error.code, crate::error::ErrorCode::LimitExceeded);
    }

    #[test]
    fn audio_mix_applies_volume_pan_and_pitch() {
        let source = AudioBuffer {
            sample_rate: 48_000,
            channels: 2,
            samples: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        };
        let track = AudioTrack {
            audio: &source,
            frame_start: 0.0,
            frame_end: None,
            frame_offset_start: 0.0,
            frame_offset_end: 0.0,
            retiming_keys: &[],
            volume: 0.5,
            pan: 1.0,
            pitch: 2.0,
        };
        let mixed = mix_audio(&[track], 0.0, 0.002, 24.0, 48_000).unwrap();
        assert_eq!(mixed.samples, vec![0.0, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn audio_mix_stops_at_strip_and_source_trim_ends() {
        let source = AudioBuffer {
            sample_rate: 48_000,
            channels: 1,
            samples: vec![0.25; 5_000],
        };
        let mut duration_limited = AudioTrack::new(&source, 0.0);
        duration_limited.frame_end = Some(1.0);
        let duration_mix = mix_audio(&[duration_limited], 0.0, 2.0, 24.0, 48_000).unwrap();
        assert!(
            duration_mix.samples[..2_000]
                .iter()
                .all(|sample| (*sample - 0.25).abs() < 1.0e-6)
        );
        assert!(
            duration_mix.samples[2_000..]
                .iter()
                .all(|sample| *sample == 0.0)
        );

        let mut trimmed = AudioTrack::new(&source, 0.0);
        trimmed.frame_offset_end = 1.0;
        let trim_mix = mix_audio(&[trimmed], 0.0, 2.0, 24.0, 48_000).unwrap();
        assert!(
            trim_mix.samples[..3_000]
                .iter()
                .all(|sample| (*sample - 0.25).abs() < 1.0e-6)
        );
        assert!(
            trim_mix.samples[3_000..]
                .iter()
                .all(|sample| *sample == 0.0)
        );
    }

    #[test]
    fn panned_mono_and_empty_audio_outputs_are_stereo() {
        let source = AudioBuffer {
            sample_rate: 48_000,
            channels: 1,
            samples: vec![0.4, 0.5],
        };
        let mut track = AudioTrack::new(&source, 0.0);
        track.pan = 1.0;
        let mixed = mix_audio(&[track], 0.0, 0.001, 24.0, 48_000).unwrap();
        assert_eq!(mixed.channels, 2);
        assert_eq!(mixed.samples, vec![0.0, 0.4, 0.0, 0.5]);

        let silence = mix_audio(&[], 0.0, 0.001, 24.0, 48_000).unwrap();
        assert_eq!(silence.channels, 2);
        assert!(silence.samples.iter().all(|sample| *sample == 0.0));
    }
}
