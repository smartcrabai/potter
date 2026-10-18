use serde::{Deserialize, Serialize};

use crate::error::{ErrorCode, PotError, Result};

mod ocio;

/// Scene-to-display view transforms, matching Blender 5.2.2's OCIO views on display `sRGB`:
/// `Standard`, `AgX` (`AgX Base Rec.1886`), `Filmic` (`Filmic sRGB`), `Raw` (`Non-Color`), and
/// `FalseColor` (`AgX False Color Rec.709`). Each view has look `None`, exposure 0, and gamma 1.
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
    /// Applies the RGB curve, exposure, and the selected view transform to scene-linear RGB. The
    /// returned channels are display-linear; the sRGB encode and gamma are in [`Self::display_encode`].
    ///
    /// Order, as Blender 5.2.2 applies it on the sRGB display with look `None` (curve mapping:
    /// `colormanagement.cc:3610-3621`; exposure `2^exposure`: `libocio_display_processor.cc:480-493`):
    /// 1. the view curve, per channel, on scene-linear values;
    /// 2. exposure;
    /// 3. the view transform:
    ///    - `Standard`: identity.
    ///    - `AgX`: E-Gamut matrix, `lg2` window, tetrahedral `AgX_Base_sRGB` LUT, Rec.1886 decode.
    ///    - `Filmic`: `lg2` window, tetrahedral `filmic_desat_33` LUT, uniform `[0, 0.66]`, 1D
    ///      `filmic_to_0-70_1-03` LUT, sRGB decode.
    ///    - `Raw`: identity.
    ///    - `FalseColor`: E-Gamut matrix, `lg2` window, tetrahedral Rec.2020 `AgX_Base` LUT, Rec.2020
    ///      luma `^(1/2.5)`, 1D `AgX_False_Color` palette, Rec.1886 decode.
    ///
    /// Errors with [`ErrorCode::UnsupportedFeature`] for any display other than `sRGB` or any look
    /// other than `none`, and with [`ErrorCode::InternalError`] if a vendored LUT fails to load.
    pub fn transform_linear(&self, rgb: [f64; 3]) -> Result<[f64; 3]> {
        self.ensure_supported()?;
        let curved = if self.curve.is_empty() {
            rgb
        } else {
            rgb.map(|value| sample_curve(value, &self.curve))
        };
        let exposure = 2.0_f64.powf(self.exposure);
        let exposed = curved.map(|value| value * exposure);
        Ok(match self.view_transform {
            ViewTransform::Standard | ViewTransform::Raw => exposed,
            ViewTransform::AgX => ocio::agx_display_linear(exposed)?,
            ViewTransform::Filmic => ocio::filmic_display_linear(exposed)?,
            ViewTransform::FalseColor => ocio::false_color_display_linear(exposed)?,
        })
    }

    /// Returns the display buffer values for `rgb`, in Blender's order (`libocio_display_processor.cc`
    /// 505-520): the sRGB OETF of [`Self::transform_linear`] (none for `Raw`), then gamma as
    /// `encoded^(1/gamma)` on the encoded value, then clamping to `[0, 1]`.
    pub fn display_encode(&self, rgb: [f64; 3]) -> Result<[f64; 3]> {
        let linear = self.transform_linear(rgb)?;
        let encoded = match self.view_transform {
            ViewTransform::Raw => linear,
            _ => linear.map(linear_to_srgb),
        };
        let graded = if self.gamma.is_finite()
            && self.gamma > 0.0
            && (self.gamma - 1.0).abs() > f64::EPSILON
        {
            encoded.map(|value| value.max(0.0).powf(1.0 / self.gamma))
        } else {
            encoded
        };
        Ok(graded.map(|value| value.clamp(0.0, 1.0)))
    }

    fn ensure_supported(&self) -> Result<()> {
        if self.display_device != "sRGB" {
            return Err(PotError::new(
                ErrorCode::UnsupportedFeature,
                format!(
                    "display device `{}` is not supported; only sRGB is implemented",
                    self.display_device
                ),
            ));
        }
        if !matches!(self.look.as_str(), "none" | "None") {
            return Err(PotError::new(
                ErrorCode::UnsupportedFeature,
                format!(
                    "color look `{}` is not supported; only `none` is implemented",
                    self.look
                ),
            ));
        }
        Ok(())
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

/// Decodes an sRGB-encoded value to linear light, the inverse of [`linear_to_srgb`].
pub(crate) fn srgb_to_linear(value: f64) -> f64 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
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

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn curve_is_applied_before_exposure() -> Result<(), Box<dyn Error>> {
        // Blender applies the curve on scene-linear input first, then exposure. Curve 0->0, 1->0.5
        // maps 0.5 to 0.25, and +1 stop doubles it to 0.5. The reverse order would give 0.5.
        let settings = ColorManagement {
            view_transform: ViewTransform::Raw,
            exposure: 1.0,
            curve: vec![[0.0, 0.0], [1.0, 0.5]],
            ..ColorManagement::default()
        };
        let out = settings.transform_linear([0.5; 3])?;
        assert!((out[0] - 0.5).abs() < 1.0e-12, "got {out:?}");
        Ok(())
    }

    #[test]
    fn gamma_is_applied_to_the_encoded_value() -> Result<(), Box<dyn Error>> {
        // Blender raises the sRGB-encoded value to 1/gamma. Gamma on linear light would differ.
        let settings = ColorManagement {
            gamma: 2.0,
            ..ColorManagement::default()
        };
        let out = settings.display_encode([0.25; 3])?;
        assert!(
            (out[0] - linear_to_srgb(0.25).sqrt()).abs() < 1.0e-12,
            "got {out:?}"
        );
        Ok(())
    }

    #[test]
    fn raw_gamma_applies_to_unencoded_values() -> Result<(), Box<dyn Error>> {
        let settings = ColorManagement {
            view_transform: ViewTransform::Raw,
            gamma: 2.0,
            ..ColorManagement::default()
        };
        let out = settings.display_encode([0.25; 3])?;
        assert!((out[0] - 0.5).abs() < 1.0e-12, "got {out:?}");
        Ok(())
    }
}
