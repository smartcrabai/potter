use serde::{Deserialize, Serialize};

use crate::error::{ErrorCode, PotError, Result};

pub mod curve;
mod looks;
mod ocio;

use looks::Look;

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
    /// Blender look token (see `Look::parse`): `none`, or a look that applies to the view.
    pub look: String,
    pub exposure: f64,
    pub gamma: f64,
    /// Blender's view curve (`view_settings.curve_mapping`), or `None` when `use_curve_mapping` is off.
    pub curve: Option<curve::CurveMapping>,
}

impl Default for ColorManagement {
    fn default() -> Self {
        Self {
            display_device: "sRGB".to_owned(),
            view_transform: ViewTransform::Standard,
            look: "none".to_owned(),
            exposure: 0.0,
            gamma: 1.0,
            curve: None,
        }
    }
}

impl ColorManagement {
    /// Returns the display values for scene-linear `rgb`, in Blender's order: the view curve, exposure
    /// `2^exposure`, the look, the view (which ends in its sRGB encode, except `Raw` and `Filmic`), then
    /// gamma `encoded^(1/gamma)` (`libocio_display_processor.cc`), clamped to `[0, 1]`.
    ///
    /// Builds the curve tables on each call. Render paths encode whole images with
    /// [`ColorManagement::display_encoder`] instead.
    pub fn display_encode(&self, rgb: [f32; 3]) -> Result<[f32; 3]> {
        self.display_encoder()?.encode(rgb)
    }

    /// Validates the settings and prepares the curve tables once, for encoding many pixels.
    ///
    /// Errors with [`ErrorCode::UnsupportedFeature`] for any display other than `sRGB`, and with
    /// [`ErrorCode::InvalidArgument`] for an unknown look or a look that does not apply to the view.
    pub fn display_encoder(&self) -> Result<DisplayEncoder> {
        if self.display_device != "sRGB" {
            return Err(PotError::new(
                ErrorCode::UnsupportedFeature,
                format!(
                    "display device `{}` is not supported; only sRGB is implemented",
                    self.display_device
                ),
            ));
        }
        let look = Look::parse(&self.look).ok_or_else(|| {
            PotError::new(
                ErrorCode::InvalidArgument,
                format!("color look `{}` is not a Blender 5.2.2 look", self.look),
            )
        })?;
        if !look.is_compatible_with(self.view_transform) {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                format!(
                    "color look `{}` does not apply to view `{:?}`",
                    self.look, self.view_transform
                ),
            ));
        }
        let curve = if let Some(curve) = &self.curve {
            curve.validate()?;
            Some(curve.prepare())
        } else {
            None
        };
        Ok(DisplayEncoder {
            view_transform: self.view_transform,
            look,
            exposure: self.exposure,
            gamma: self.gamma,
            curve,
        })
    }
}

/// Display encoding with its curve tables prepared once, for encoding many pixels.
pub struct DisplayEncoder {
    view_transform: ViewTransform,
    look: Look,
    exposure: f64,
    gamma: f64,
    curve: Option<curve::PreparedCurve>,
}

impl DisplayEncoder {
    /// Encodes scene-linear `rgb` for the display; see [`ColorManagement::display_encode`].
    pub fn encode(&self, rgb: [f32; 3]) -> Result<[f32; 3]> {
        let curved = self
            .curve
            .as_ref()
            .map_or(rgb, |view_curve| view_curve.evaluate_rgb(rgb));
        let encoded = ocio::display_view(self.view_transform, self.look, self.exposure, curved)?;
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

/// Cycles' sRGB-to-linear decode for an image texel, in f32, as its SSE2 and NEON kernels compute it
/// (`color_srgb_to_linear_sse2`, Cycles `util/color.h`).
///
/// The approximation differs from [`srgb_to_linear`] by at most about 5e-7 over [0, 1], and Cycles' kernels use
/// it for image textures, so the image sampler uses this decode. `mul_add` is the fused `madd` of the arm64 build
/// (NEON `vfmaq_f32`), which Blender's arm64 and this crate share; the x86 SSE2 build rounds the multiply and the
/// add separately.
#[must_use]
pub fn cycles_srgb_to_linear(encoded: f32) -> f32 {
    if encoded < 0.04045_f32 {
        (encoded * (1.0_f32 / 12.92_f32)).max(0.0)
    } else {
        cycles_fastpow24((encoded + 0.055_f32) * (1.0_f32 / 1.055_f32))
    }
}

/// Cycles' `fastpow24_sse2`: x^2.4 from a bit-pattern guess for x^(4/5), three Newton steps on the fifth root of
/// x^4, then x^(12/5) = y · y · y.
fn cycles_fastpow24(arg: f32) -> f32 {
    // `fastpow_sse2<0x3F4CCCCD, 0x4F55A7FB>`: scale by 2^(127/(4/5) - 127) · 0.994^(5/4), reinterpret the bits as an
    // integer, scale by 4/5, and round back to bits.
    let scaled = arg * f32::from_bits(0x4F55_A7FB);
    let scaled = int_bits_as_f32(scaled.to_bits());
    let scaled = scaled * f32::from_bits(0x3F4C_CCCD);
    let mut root = f32::from_bits(round_to_bits(scaled));
    let arg2 = arg * arg;
    let arg4 = arg2 * arg2;
    root = improve_fifth_root(root, arg4);
    root = improve_fifth_root(root, arg4);
    root = improve_fifth_root(root, arg4);
    root * (root * root)
}

/// One Newton step for the fifth root of `arg4` (Cycles' `improve_5throot_solution_sse2`).
fn improve_fifth_root(old: f32, arg4: f32) -> f32 {
    let approx2 = old * old;
    let approx4 = approx2 * approx2;
    let quotient = arg4 / approx4;
    let sum = 4.0_f32.mul_add(old, quotient);
    sum * (1.0_f32 / 5.0_f32)
}

/// `cast(make_float4(int4))`: the integer whose bits are `bits`, converted to float with rounding to nearest, as
/// `_mm_cvtepi32_ps` does.
#[expect(
    clippy::cast_precision_loss,
    reason = "the integer-to-float conversion rounds to nearest, as _mm_cvtepi32_ps does"
)]
fn int_bits_as_f32(bits: u32) -> f32 {
    i32::from_ne_bytes(bits.to_ne_bytes()) as f32
}

/// `make_int4(float4)` (`_mm_cvtps_epi32`), returning the integer's bits: round to nearest, ties to even, and
/// out-of-range values give `i32::MIN`, as the hardware does.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the rounded value is checked to lie inside the i32 range before the conversion"
)]
fn round_to_bits(value: f32) -> u32 {
    let rounded = value.round_ties_even();
    let integer = if (-2_147_483_648.0..2_147_483_648.0).contains(&rounded) {
        rounded as i32
    } else {
        i32::MIN
    };
    u32::from_ne_bytes(integer.to_ne_bytes())
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
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::Standard,
            input: [0.05, 0.05, 0.05],
            output: [0x3e7d_bf6e, 0x3e7d_bf6e, 0x3e7d_bf6e],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::Standard,
            input: [0.18, 0.18, 0.18],
            output: [0x3eec_36ac, 0x3eec_36ac, 0x3eec_36ac],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::Standard,
            input: [4.5, 0.0, 1.1],
            output: [0x3f80_0000, 0x0000_0000, 0x3f80_0000],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::Standard,
            input: [0.02, 0.5, 4.0],
            output: [0x3e1b_583c, 0x3f3c_406d, 0x3f80_0000],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::AgX,
            input: [0.05, 0.05, 0.05],
            output: [0x3e68_a8e6, 0x3e68_a8a0, 0x3e68_aeb2],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::AgX,
            input: [0.18, 0.18, 0.18],
            output: [0x3eec_322a, 0x3eec_31ec, 0x3eec_36aa],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::AgX,
            input: [4.5, 0.0, 1.1],
            output: [0x3f7c_1ed7, 0x3f2d_5cf2, 0x3f46_7a7b],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::AgX,
            input: [0.02, 0.5, 4.0],
            output: [0x3f19_190e, 0x3f39_6e41, 0x3f74_2c65],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::Filmic,
            input: [0.05, 0.05, 0.05],
            output: [0x3e8b_d0d7, 0x3e8b_d0d7, 0x3e8b_d0d7],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::Filmic,
            input: [0.18, 0.18, 0.18],
            output: [0x3f00_0dd9, 0x3f00_0dd9, 0x3f00_0dd9],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::Filmic,
            input: [4.5, 0.0, 1.1],
            output: [0x3f74_e3f9, 0x3777_f0f8, 0x3f52_1541],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::Filmic,
            input: [0.02, 0.5, 4.0],
            output: [0x3e23_093e, 0x3f31_e876, 0x3f73_0add],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::Raw,
            input: [0.05, 0.05, 0.05],
            output: [0x3d4c_cccd, 0x3d4c_cccd, 0x3d4c_cccd],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::Raw,
            input: [0.18, 0.18, 0.18],
            output: [0x3e38_51ec, 0x3e38_51ec, 0x3e38_51ec],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::Raw,
            input: [4.5, 0.0, 1.1],
            output: [0x3f80_0000, 0x0000_0000, 0x3f80_0000],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::Raw,
            input: [0.02, 0.5, 4.0],
            output: [0x3ca3_d70a, 0x3f00_0000, 0x3f80_0000],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::FalseColor,
            input: [0.05, 0.05, 0.05],
            output: [0x0000_0000, 0x3f80_0000, 0x3f80_0000],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::FalseColor,
            input: [0.18, 0.18, 0.18],
            output: [0x3ef1_ec06, 0x3ef1_ec06, 0x3ef1_ec06],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::FalseColor,
            input: [4.5, 0.0, 1.1],
            output: [0x3f80_0000, 0x3f1e_3760, 0x0000_0000],
        },
        Vector {
            exposure: 0.0,
            gamma: 1.0,
            view: ViewTransform::FalseColor,
            input: [0.02, 0.5, 4.0],
            output: [0x3f80_0000, 0x3f80_0000, 0x0000_0000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Standard,
            input: [0.05, 0.05, 0.05],
            output: [0x3f11_0cc6, 0x3f11_0cc6, 0x3f11_0cc6],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Standard,
            input: [0.18, 0.18, 0.18],
            output: [0x3f44_02b6, 0x3f44_02b6, 0x3f44_02b6],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Standard,
            input: [4.5, 0.0, 1.1],
            output: [0x3f80_0000, 0x0000_0000, 0x3f80_0000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Standard,
            input: [0.02, 0.5, 4.0],
            output: [0x3ee5_f154, 0x3f76_6a32, 0x3f80_0000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::AgX,
            input: [0.05, 0.05, 0.05],
            output: [0x3f0e_1678, 0x3f0e_1662, 0x3f0e_181c],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::AgX,
            input: [0.18, 0.18, 0.18],
            output: [0x3f41_9908, 0x3f41_98f6, 0x3f41_9a68],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::AgX,
            input: [4.5, 0.0, 1.1],
            output: [0x3f7f_d186, 0x3f5e_fb65, 0x3f6a_466d],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::AgX,
            input: [0.02, 0.5, 4.0],
            output: [0x3f55_6099, 0x3f64_3692, 0x3f7e_2f69],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Filmic,
            input: [0.05, 0.05, 0.05],
            output: [0x3f19_0bac, 0x3f19_0bac, 0x3f19_0bac],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Filmic,
            input: [0.18, 0.18, 0.18],
            output: [0x3f46_848c, 0x3f46_848c, 0x3f46_848c],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Filmic,
            input: [4.5, 0.0, 1.1],
            output: [0x3f7d_93ca, 0x3b7b_efef, 0x3f70_b71e],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Filmic,
            input: [0.02, 0.5, 4.0],
            output: [0x3eef_3314, 0x3f62_506f, 0x3f7c_fb5c],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Raw,
            input: [0.05, 0.05, 0.05],
            output: [0x3e94_7894, 0x3e94_7894, 0x3e94_7894],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Raw,
            input: [0.18, 0.18, 0.18],
            output: [0x3f0c_da28, 0x3f0c_da28, 0x3f0c_da28],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Raw,
            input: [4.5, 0.0, 1.1],
            output: [0x3f80_0000, 0x0000_0000, 0x3f80_0000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::Raw,
            input: [0.02, 0.5, 4.0],
            output: [0x3e3b_cd22, 0x3f6a_c124, 0x3f80_0000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::FalseColor,
            input: [0.05, 0.05, 0.05],
            output: [0x0000_0000, 0x3f80_0000, 0x3f2f_f8a0],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::FalseColor,
            input: [0.18, 0.18, 0.18],
            output: [0x3f2f_f8a0, 0x3f80_0000, 0x0000_0000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::FalseColor,
            input: [4.5, 0.0, 1.1],
            output: [0x3f80_0000, 0x3f2f_f8a0, 0x0000_0000],
        },
        Vector {
            exposure: 0.75,
            gamma: 2.0,
            view: ViewTransform::FalseColor,
            input: [0.02, 0.5, 4.0],
            output: [0x3f80_0000, 0x3f2f_f8a0, 0x0000_0000],
        },
    ];

    #[test]
    fn curve_is_applied_before_exposure() -> Result<(), Box<dyn Error>> {
        // The curve maps (0, 0.5, 1) to (0, 0.1, 1). Exposure -1 halves values. Curve first gives
        // curve(1.0) * 0.5 = 0.5; exposure first would give curve(0.5) = 0.1.
        let mut view_curve = curve::CurveMapping {
            extend_extrapolate: false,
            ..curve::CurveMapping::default()
        };
        view_curve.curves[3].points = [(0.0_f32, 0.0_f32), (0.5, 0.1), (1.0, 1.0)]
            .map(|(x, y)| curve::CurvePoint {
                location: [x, y],
                handle: curve::HandleType::Vector,
            })
            .to_vec();
        let settings = ColorManagement {
            view_transform: ViewTransform::Raw,
            exposure: -1.0,
            curve: Some(view_curve),
            ..ColorManagement::default()
        };
        let out = settings.display_encode([1.0; 3])?;
        assert!((out[0] - 0.5).abs() < 1.0e-3, "got {out:?}");
        Ok(())
    }

    #[test]
    fn display_encoder_rejects_deserialized_malformed_curve_before_preparing()
    -> std::result::Result<(), Box<dyn Error>> {
        let mut value = serde_json::to_value(ColorManagement::default())?;
        value["curve"] = serde_json::to_value(curve::CurveMapping::default())?;
        value["curve"]["curves"][0]["points"] = serde_json::json!([{
            "location": [0.0, 0.0],
            "handle": "auto"
        }]);
        let settings: ColorManagement = serde_json::from_value(value)?;

        let Err(error) = settings.display_encoder() else {
            return Err("invalid curve was accepted".into());
        };
        assert_eq!(error.code, crate::error::ErrorCode::InvalidArgument);
        assert!(error.message.contains("curve 0 needs at least two points"));
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
        // OCIO's approximate pow returns 0.5000035 here (about 7e-6 relative), not libm's 0.5.
        assert!((out[0] - 0.5).abs() < 1.0e-5, "got {out:?}");
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
            for (channel, &expected) in vector.output.iter().enumerate() {
                assert_eq!(
                    output[channel].to_bits(),
                    expected,
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
