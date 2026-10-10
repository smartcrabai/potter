use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewTransform {
    #[default]
    Standard,
    AgX,
    Filmic,
    Raw,
    FalseColor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ColorManagement {
    pub display_device: String,
    pub view_transform: ViewTransform,
    pub look: String,
    pub exposure: f64,
    pub gamma: f64,
    /// Shared RGB display curve control points `(input, output)`, sorted by input.
    pub curve: Vec<[f64; 2]>,
}

impl Default for ColorManagement {
    fn default() -> Self {
        Self {
            display_device: "sRGB".to_owned(),
            view_transform: ViewTransform::Standard,
            look: "none".to_owned(),
            exposure: 0.0,
            gamma: 1.0,
            curve: Vec::new(),
        }
    }
}

impl ColorManagement {
    /// Applies exposure, the selected scene-linear view transform, look, curve, and gamma.
    /// The returned channels are display-linear values; sRGB encoding is a separate step.
    #[must_use]
    pub fn transform_linear(&self, rgb: [f64; 3]) -> [f64; 3] {
        let exposure = 2.0_f64.powf(self.exposure);
        let exposed = rgb.map(|value| value * exposure);
        let mut transformed = match self.view_transform {
            ViewTransform::Standard | ViewTransform::Raw => exposed,
            ViewTransform::AgX => agx(exposed),
            ViewTransform::Filmic => exposed.map(filmic_lut_approximation),
            ViewTransform::FalseColor => false_color(exposed),
        };
        apply_look(&mut transformed, &self.look);
        if !self.curve.is_empty() {
            transformed = transformed.map(|value| sample_curve(value, &self.curve));
        }
        if self.gamma.is_finite() && self.gamma > 0.0 && (self.gamma - 1.0).abs() > f64::EPSILON {
            transformed = transformed.map(|value| value.max(0.0).powf(1.0 / self.gamma));
        }
        transformed
    }
}

/// Converts linear display values to the standard sRGB transfer function.
#[must_use]
pub fn linear_to_srgb(value: f64) -> f64 {
    linear_to_srgb_impl(value, true)
}

pub(crate) fn linear_to_srgb_unclamped(value: f64) -> f64 {
    linear_to_srgb_impl(value, false)
}

fn linear_to_srgb_impl(value: f64, clamp_high_branch: bool) -> f64 {
    if value <= 0.003_130_8 {
        12.92 * value
    } else {
        let value = if clamp_high_branch {
            value.max(0.0)
        } else {
            value
        };
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// Converts linear `f32` values to sRGB without changing intermediate precision.
#[must_use]
pub(crate) fn linear_to_srgb_f32(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.max(0.0).powf(1.0 / 2.4) - 0.055
    }
}

fn agx(rgb: [f64; 3]) -> [f64; 3] {
    // AgX's published color-space matrix family, followed by a smooth sigmoid fit to its
    // log-domain look-up curve. This is a compact monotone fit, not a bit-identical OCIO LUT.
    const INPUT: [[f64; 3]; 3] = [
        [0.842_479, 0.042_328, 0.042_375],
        [0.078_434, 0.878_468, 0.078_433],
        [0.079_223, 0.079_166, 0.879_142],
    ];
    let working = INPUT.map(|row| row[0] * rgb[0] + row[1] * rgb[1] + row[2] * rgb[2]);
    working.map(|value| {
        let value = value.max(0.0);
        let log_value = (value + 0.000_001).log2();
        let normalized = ((log_value + 10.0) / 16.0).clamp(0.0, 1.0);

        normalized * normalized * (3.0 - 2.0 * normalized)
    })
}

/// Analytic approximation of the familiar Filmic display LUT (Hable/Uncharted-2 curve).
/// The normalized rational curve is used in place of a sampled Blender Filmic LUT.
fn filmic_lut_approximation(value: f64) -> f64 {
    const A: f64 = 0.15;
    const B: f64 = 0.50;
    const C: f64 = 0.10;
    const D: f64 = 0.20;
    const E: f64 = 0.02;
    const F: f64 = 0.30;
    let curve = |x: f64| ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    (curve(value.max(0.0)) / curve(11.2)).clamp(0.0, 1.0)
}

fn false_color(rgb: [f64; 3]) -> [f64; 3] {
    let luminance = (rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722).max(0.0);
    let normalized = luminance / (1.0 + luminance);
    let stops = [
        [0.0, 0.0, 0.5],
        [0.0, 0.8, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ];
    let scaled = normalized * 4.0;
    let index = (scaled.floor() as usize).min(3);
    let fraction = scaled - index as f64;
    std::array::from_fn(|channel| {
        stops[index][channel] * (1.0 - fraction) + stops[index + 1][channel] * fraction
    })
}

fn apply_look(rgb: &mut [f64; 3], look: &str) {
    let contrast: f64 = match look {
        "high_contrast" | "High Contrast" | "AgX - High Contrast" | "AgX - Punchy" => 1.25,
        "medium_high_contrast" | "Medium High Contrast" | "AgX - Medium High Contrast" => 1.1,
        "medium_low_contrast" | "Medium Low Contrast" | "AgX - Medium Low Contrast" => 0.9,
        "low_contrast" | "Low Contrast" | "AgX - Low Contrast" => 0.75,
        "very_low_contrast" | "Very Low Contrast" | "AgX - Very Low Contrast" => 0.6,
        "very_high_contrast" | "Very High Contrast" | "AgX - Very High Contrast" => 1.4,
        _ => 1.0,
    };
    if (contrast - 1.0).abs() > f64::EPSILON {
        for value in rgb {
            *value = (*value - 0.18).mul_add(contrast, 0.18);
        }
    }
}

pub(crate) fn sample_curve(value: f64, curve: &[[f64; 2]]) -> f64 {
    let first = curve[0];
    if value <= first[0] {
        return first[1];
    }
    for pair in curve.windows(2) {
        let [left, right] = [pair[0], pair[1]];
        if value <= right[0] {
            let width = right[0] - left[0];
            if width <= 0.0 {
                return right[1];
            }
            let factor = (value - left[0]) / width;
            return interpolate_curve_values(left[1], right[1], factor);
        }
    }
    curve[curve.len() - 1][1]
}

pub(crate) fn interpolate_curve_values(left: f64, right: f64, factor: f64) -> f64 {
    left * (1.0 - factor) + right * factor
}

#[cfg(kani)]
mod verification {
    #[kani::proof]
    fn srgb_transfer_preserves_zero() {
        assert!(super::linear_to_srgb(0.0).abs() < f64::EPSILON);
    }
}
