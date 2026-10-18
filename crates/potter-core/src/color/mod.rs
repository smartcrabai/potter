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
    /// Returns the display values for scene-linear `rgb`, in Blender's order: the RGB curve on
    /// scene-linear values (`colormanagement.cc`), exposure `2^exposure`, the view (which ends in its
    /// sRGB encode, except `Raw`), then gamma `encoded^(1/gamma)` on the encoded value
    /// (`libocio_display_processor.cc`), clamped to `[0, 1]`. The arithmetic is float32 and follows
    /// OCIO's fast paths; see `ocio` for the operation order.
    ///
    /// Errors with [`ErrorCode::UnsupportedFeature`] for any display other than `sRGB` or any look
    /// other than `none`, and with [`ErrorCode::InternalError`] if a vendored LUT fails to load.
    pub fn display_encode(&self, rgb: [f32; 3]) -> Result<[f32; 3]> {
        self.ensure_supported()?;
        let curved = if self.curve.is_empty() {
            rgb
        } else {
            rgb.map(|value| sample_curve(f64::from(value), &self.curve) as f32)
        };
        let encoded = ocio::display_view(self.view_transform, self.exposure, curved)?;
        let graded = if self.gamma.is_finite()
            && self.gamma > 0.0
            && (self.gamma - 1.0).abs() > f64::EPSILON
        {
            let exponent = (1.0 / self.gamma) as f32;
            encoded.map(|value| ocio::fast_pow(value, exponent))
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

    /// One committed float32 emulation vector: the display output of `view` after `exposure` and `gamma`
    /// for `input`, as bit patterns. The vectors come from the numpy emulation of OCIO's default
    /// processor on macOS arm64 (fused where that build fuses), not from a run of OCIO itself.
    struct Vector {
        exposure: f64,
        gamma: f64,
        view: ViewTransform,
        input: [f32; 3],
        output: [u32; 3],
    }

    const VECTORS: [Vector; 40] = [
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::Standard,
            input: [0.05, 0.05, 0.05],
            output: [0x3e7dbf6e, 0x3e7dbf6e, 0x3e7dbf6e],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::Standard,
            input: [0.18, 0.18, 0.18],
            output: [0x3eec36ac, 0x3eec36ac, 0x3eec36ac],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::Standard,
            input: [4.5, 0.0, 1.1],
            output: [0x3f800000, 0x00000000, 0x3f800000],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::Standard,
            input: [0.02, 0.5, 4.0],
            output: [0x3e1b583c, 0x3f3c406d, 0x3f800000],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::AgX,
            input: [0.05, 0.05, 0.05],
            output: [0x3e68a8e6, 0x3e68a8a0, 0x3e68aeb2],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::AgX,
            input: [0.18, 0.18, 0.18],
            output: [0x3eec322a, 0x3eec31ec, 0x3eec36aa],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::AgX,
            input: [4.5, 0.0, 1.1],
            output: [0x3f7c1ed7, 0x3f2d5cf2, 0x3f467a7b],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::AgX,
            input: [0.02, 0.5, 4.0],
            output: [0x3f19190e, 0x3f396e41, 0x3f742c65],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::Filmic,
            input: [0.05, 0.05, 0.05],
            output: [0x3e8bd0d7, 0x3e8bd0d7, 0x3e8bd0d7],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::Filmic,
            input: [0.18, 0.18, 0.18],
            output: [0x3f000dd9, 0x3f000dd9, 0x3f000dd9],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::Filmic,
            input: [4.5, 0.0, 1.1],
            output: [0x3f74e3f9, 0x3777f0f8, 0x3f521541],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::Filmic,
            input: [0.02, 0.5, 4.0],
            output: [0x3e23093e, 0x3f31e876, 0x3f730add],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::Raw,
            input: [0.05, 0.05, 0.05],
            output: [0x3d4ccccd, 0x3d4ccccd, 0x3d4ccccd],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::Raw,
            input: [0.18, 0.18, 0.18],
            output: [0x3e3851ec, 0x3e3851ec, 0x3e3851ec],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::Raw,
            input: [4.5, 0.0, 1.1],
            output: [0x3f800000, 0x00000000, 0x3f800000],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::Raw,
            input: [0.02, 0.5, 4.0],
            output: [0x3ca3d70a, 0x3f000000, 0x3f800000],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::FalseColor,
            input: [0.05, 0.05, 0.05],
            output: [0x00000000, 0x3f800000, 0x3f800000],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::FalseColor,
            input: [0.18, 0.18, 0.18],
            output: [0x3ef1ec06, 0x3ef1ec06, 0x3ef1ec06],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::FalseColor,
            input: [4.5, 0.0, 1.1],
            output: [0x3f800000, 0x3f1e3760, 0x00000000],
        },
        Vector {
            exposure: 0.00,
            gamma: 1.0,
            view: ViewTransform::FalseColor,
            input: [0.02, 0.5, 4.0],
            output: [0x3f800000, 0x3f800000, 0x00000000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Standard,
            input: [0.05, 0.05, 0.05],
            output: [0x3f110cc6, 0x3f110cc6, 0x3f110cc6],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Standard,
            input: [0.18, 0.18, 0.18],
            output: [0x3f4402b6, 0x3f4402b6, 0x3f4402b6],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Standard,
            input: [4.5, 0.0, 1.1],
            output: [0x3f800000, 0x00000000, 0x3f800000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Standard,
            input: [0.02, 0.5, 4.0],
            output: [0x3ee5f154, 0x3f766a32, 0x3f800000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::AgX,
            input: [0.05, 0.05, 0.05],
            output: [0x3f0e1678, 0x3f0e1662, 0x3f0e181c],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::AgX,
            input: [0.18, 0.18, 0.18],
            output: [0x3f419908, 0x3f4198f6, 0x3f419a68],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::AgX,
            input: [4.5, 0.0, 1.1],
            output: [0x3f7fd186, 0x3f5efb65, 0x3f6a466d],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::AgX,
            input: [0.02, 0.5, 4.0],
            output: [0x3f556099, 0x3f643692, 0x3f7e2f69],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Filmic,
            input: [0.05, 0.05, 0.05],
            output: [0x3f190bac, 0x3f190bac, 0x3f190bac],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Filmic,
            input: [0.18, 0.18, 0.18],
            output: [0x3f46848c, 0x3f46848c, 0x3f46848c],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Filmic,
            input: [4.5, 0.0, 1.1],
            output: [0x3f7d93ca, 0x3b7befef, 0x3f70b71e],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Filmic,
            input: [0.02, 0.5, 4.0],
            output: [0x3eef3314, 0x3f62506f, 0x3f7cfb5c],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Raw,
            input: [0.05, 0.05, 0.05],
            output: [0x3e947894, 0x3e947894, 0x3e947894],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Raw,
            input: [0.18, 0.18, 0.18],
            output: [0x3f0cda28, 0x3f0cda28, 0x3f0cda28],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Raw,
            input: [4.5, 0.0, 1.1],
            output: [0x3f800000, 0x00000000, 0x3f800000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Raw,
            input: [0.02, 0.5, 4.0],
            output: [0x3e3bcd22, 0x3f6ac124, 0x3f800000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::FalseColor,
            input: [0.05, 0.05, 0.05],
            output: [0x00000000, 0x3f800000, 0x3f2ff8a0],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::FalseColor,
            input: [0.18, 0.18, 0.18],
            output: [0x3f2ff8a0, 0x3f800000, 0x00000000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::FalseColor,
            input: [4.5, 0.0, 1.1],
            output: [0x3f800000, 0x3f2ff8a0, 0x00000000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::FalseColor,
            input: [0.02, 0.5, 4.0],
            output: [0x3f800000, 0x3f2ff8a0, 0x00000000],
        },
    ];

    #[test]
    fn curve_is_applied_before_exposure() -> Result<(), Box<dyn Error>> {
        // The curve maps scene-linear 0.5 to 0.25 (curve 0->0, 1->0.5); +1 stop doubles it to 0.5.
        // Applying exposure first would give curve(1.0) = 0.5.
        let settings = ColorManagement {
            view_transform: ViewTransform::Raw,
            exposure: 1.0,
            curve: vec![[0.0, 0.0], [1.0, 0.5]],
            ..ColorManagement::default()
        };
        let out = settings.display_encode([0.5; 3])?;
        assert!((out[0] - 0.5).abs() < 1.0e-6, "got {out:?}");
        Ok(())
    }

    #[test]
    fn gamma_is_applied_to_the_encoded_value() -> Result<(), Box<dyn Error>> {
        // Gamma raises the sRGB-encoded value to 1/gamma; on linear light the result would differ.
        let settings = ColorManagement {
            gamma: 2.0,
            ..ColorManagement::default()
        };
        let out = settings.display_encode([0.25; 3])?;
        assert!(
            (out[0] - linear_to_srgb(0.25).sqrt() as f32).abs() < 1.0e-5,
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
        assert!((out[0] - 0.5).abs() < 1.0e-6, "got {out:?}");
        Ok(())
    }

    #[test]
    fn display_encode_matches_committed_f32_emulation_vectors() {
        for vector in VECTORS {
            let settings = ColorManagement {
                view_transform: vector.view,
                exposure: vector.exposure,
                gamma: vector.gamma,
                ..ColorManagement::default()
            };
            let output = settings
                .display_encode(vector.input)
                .unwrap_or_else(|error| panic!("display_encode failed: {error}"));
            for channel in 0..3 {
                assert_eq!(
                    output[channel].to_bits(),
                    vector.output[channel],
                    "{:?} exposure {} gamma {} input {:?} channel {channel}",
                    vector.view,
                    vector.exposure,
                    vector.gamma,
                    vector.input,
                );
            }
        }
    }
}
