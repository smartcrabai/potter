//! Blender 5.2.2 `OpenColorIO` display views for the `sRGB` display, in float32.
//!
//! Each view reproduces the operation order of `OpenColorIO` 2.5.0's default CPU processor (the one
//! Blender uses) on arm64 with the `FAST_LOG_EXP_POW` optimisation, which is part of the default
//! optimisation level:
//! - `pow` and `log2` use OCIO's SSE polynomial approximations (`SSE.h`: `sseLog2`, `sseExp2`,
//!   `ssePower`), which are not libm results. Their arithmetic is plain float32 with separate
//!   rounding for each operation.
//! - The `lg2` allocation, the 3D tetrahedral LUT and the 1D LUT lerp use one fused multiply-add
//!   where Blender's arm64 scalar code contracts `a * b + c` (`mul_add` here). The 3x3 colour matrix
//!   and the sRGB encode use separate rounding, as the SSE intrinsics do.
//! - Exposure is folded into the matrix in double precision, then rounded to float32 once.
//!
//! The FMA choice is the one measured against Blender's default processor on macOS arm64. Linux
//! `x86_64` may round some of these operations differently; the bound in the Blender oracle test covers
//! that (see `tests/color_view_transform_blender.rs`).
//!
//! The LUTs in `color/luts/` are byte-for-byte copies of Blender 5.2.2 data. Source, SHA-256,
//! authors, and licence status are recorded in `color/luts/NOTICE`.

use std::sync::LazyLock;

use super::ViewTransform;
use crate::error::{ErrorCode, PotError, Result};

const AGX_BASE_SRGB_CUBE: &str = include_str!("luts/AgX_Base_sRGB.cube");
const AGX_BASE_REC2020_CUBE: &str = include_str!("luts/AgX_Base_Rec2020.cube");
const AGX_FALSE_COLOR_SPI1D: &str = include_str!("luts/AgX_False_Color.spi1d");
const FILMIC_DESAT_33_CUBE: &str = include_str!("luts/filmic_desat_33.cube");
const FILMIC_TO_0_70_SPI1D: &str = include_str!("luts/filmic_to_0-70_1-03.spi1d");

static AGX_BASE_SRGB: LazyLock<Result<Lut3d, String>> =
    LazyLock::new(|| parse_cube(AGX_BASE_SRGB_CUBE));
static AGX_BASE_REC2020: LazyLock<Result<Lut3d, String>> =
    LazyLock::new(|| parse_cube(AGX_BASE_REC2020_CUBE));
static AGX_FALSE_COLOR: LazyLock<Result<Lut1d, String>> =
    LazyLock::new(|| parse_spi1d(AGX_FALSE_COLOR_SPI1D));
static FILMIC_DESAT_33: LazyLock<Result<Lut3d, String>> =
    LazyLock::new(|| parse_cube(FILMIC_DESAT_33_CUBE));
static FILMIC_TO_0_70: LazyLock<Result<Lut1d, String>> =
    LazyLock::new(|| parse_spi1d(FILMIC_TO_0_70_SPI1D));

/// `lg2` allocation window of the AgX views (`AgX Base Rec.1886`, `AgX False Color Rec.709`).
const AGX_LOG2_MIN: f64 = -12.473_93;
const AGX_LOG2_MAX: f64 = 12.526_068_811_7;
/// `lg2` allocation window of the Filmic view (`Filmic sRGB`).
const FILMIC_LOG2_MIN: f64 = -12.473_931_188;
const FILMIC_LOG2_MAX: f64 = 12.526_068_812;
/// `uniform` allocation upper bound of the Filmic desaturation LUT output.
const FILMIC_UNIFORM_MAX: f64 = 0.66;

const FALSE_COLOR_EXPONENT: f64 = 2.5;

/// Rec.709 linear to `FilmLight` E-Gamut linear: `XYZ(D65)->E-Gamut * inverse(XYZ(D65)->Rec.709)`,
/// computed numerically from the Blender 5.2.2 config matrices `XYZ (D65) -> Linear Rec.709` and
/// `XYZ (D65) -> Linear FilmLight E-Gamut`, rounded to 12 decimals.
const REC709_TO_E_GAMUT: [[f64; 3]; 3] = [
    [0.559_371_133_310, 0.304_783_346_623, 0.135_845_560_480],
    [0.076_220_705_484, 0.787_971_770_103, 0.135_807_466_175],
    [0.065_526_711_770, 0.164_546_754_604, 0.769_926_501_650],
];

/// Luma weights of the `AgX False Color Rec.709` view, applied to Rec.2020 linear values.
const FALSE_COLOR_LUMA: [f64; 3] = [
    0.258_923_535_568_984_8,
    0.610_498_534_606_652_5,
    0.130_577_929_824_362_84,
];

/// Exponent of the Rec.1886 decode (`AgX`, `FalseColor`) and of the gamma stage's inverse pair.
const DECODE_EXPONENT: f32 = 2.4;

/// OCIO's `FLT_MIN` clamp before `log2` (`LogRendererSSE`).
const FLT_MIN: f32 = f32::MIN_POSITIVE;

// OCIO SSE.h constants: degree-5 minimax log2 on [1, 2) and degree-4 exp2 on [0, 1), as float32.
const PNLOG5: f32 = f32::from_bits(0x3d37_cd65);
const PNLOG4: f32 = f32::from_bits(0xbed5_47d7);
const PNLOG3: f32 = f32::from_bits(0x3fd0_c97c);
const PNLOG2: f32 = f32::from_bits(0xc063_4031);
const PNLOG1: f32 = f32::from_bits(0x40a2_ef4c);
const PNLOG0: f32 = f32::from_bits(0xc033_392a);
const PNEXP4: f32 = f32::from_bits(0x3c5d_be6a);
const PNEXP3: f32 = f32::from_bits(0x3d55_09f8);
const PNEXP2: f32 = f32::from_bits(0x3e77_3cc5);
const PNEXP1: f32 = f32::from_bits(0x3f31_68b3);
const PNEXP0: f32 = f32::from_bits(0x3f80_0016);

const EXP_MASK: u32 = 0x7F80_0000;
const SIGN_MANTISSA_MASK: u32 = 0x807F_FFFF;
const ONE_BITS: u32 = 0x3F80_0000;

// MONCURVE reverse (`sRGB` encode) parameters, float32 as OCIO's `ComputeParamsRev` stores them.
const ENCODE_GAMMA: f32 = f32::from_bits(0x3ed5_5555);
const ENCODE_OFFSET: f32 = f32::from_bits(0x3d61_47ae);
const ENCODE_BREAK: f32 = f32::from_bits(0x3b47_39a4);
const ENCODE_SLOPE: f32 = f32::from_bits(0x414e_c578);
const ENCODE_SCALE: f32 = f32::from_bits(0x3f87_0a3d);

/// Float32 value of `x` with OCIO's `cvttps` truncation: NaN and out-of-range inputs give `i32::MIN`.
fn truncate_to_i32(x: f32) -> i32 {
    if x.is_nan() || x >= 2_147_483_648.0 || x < -2_147_483_648.0 {
        i32::MIN
    } else {
        x.trunc() as i32
    }
}

/// OCIO `sseLog2`: the mantissa in `[1, 2)` through the degree-5 polynomial, plus the exponent.
fn fast_log2(x: f32) -> f32 {
    let bits = x.to_bits();
    let mantissa = f32::from_bits((bits & SIGN_MANTISSA_MASK) | ONE_BITS);
    let mut poly = PNLOG5 * mantissa;
    poly = (poly + PNLOG4) * mantissa;
    poly = (poly + PNLOG3) * mantissa;
    poly = (poly + PNLOG2) * mantissa;
    poly = (poly + PNLOG1) * mantissa;
    poly += PNLOG0;
    let exponent = ((bits & EXP_MASK) >> 23) as i32 - 127;
    poly + exponent as f32
}

/// OCIO `sseExp2`: `floor` and fraction split, the degree-4 polynomial, and `2^floor` by bits.
fn fast_exp2(x: f32) -> f32 {
    // `cmpnle(0, x)`: all ones for negative or NaN inputs, which subtracts one from the truncation.
    let below: i32 = if x < 0.0 || x.is_nan() { -1 } else { 0 };
    let floor_x = truncate_to_i32(x).wrapping_add(below);
    let scale = f32::from_bits(((floor_x.wrapping_add(127)) as u32) << 23);
    let fraction = x - floor_x as f32;
    let mut poly = PNEXP4 * fraction;
    poly = (poly + PNEXP3) * fraction;
    poly = (poly + PNEXP2) * fraction;
    poly = (poly + PNEXP1) * fraction;
    poly += PNEXP0;
    let mut result = scale * poly;
    if x < -126.0 {
        result = 0.0;
    }
    if x >= 128.0 {
        result = f32::INFINITY;
    }
    result
}

/// OCIO `ssePower`: `exp2(g * log2(x))`, zero for non-positive and NaN bases.
pub(super) fn fast_pow(x: f32, exponent: f32) -> f32 {
    let value = fast_exp2(exponent * fast_log2(x));
    if x > 0.0 { value } else { 0.0 }
}

/// Affine `lg2` allocation fit: `fma(v, slope, offset)` with `slope` and `offset` rounded to float32.
fn fit_lg2(value: f32, old_min: f64, old_max: f64) -> f32 {
    let slope = (1.0 / (old_max - old_min)) as f32;
    let offset = (-old_min) as f32 * slope;
    value.mul_add(slope, offset)
}

/// `lg2` allocation forward (`LogRendererSSE`, then the fit), per channel.
fn lg2_position(rgb: [f32; 3], old_min: f64, old_max: f64) -> [f32; 3] {
    rgb.map(|value| fit_lg2(fast_log2(value.max(FLT_MIN)), old_min, old_max))
}

/// Three-by-three colour matrix in the SSE order: `(r*m0 + g*m1) + b*m2`, each product rounded.
fn mat3(rgb: [f32; 3], matrix: &[[f32; 3]; 3]) -> [f32; 3] {
    std::array::from_fn(|row| {
        let p0 = rgb[0] * matrix[row][0];
        let p1 = rgb[1] * matrix[row][1];
        let p2 = rgb[2] * matrix[row][2];
        (p0 + p1) + p2
    })
}

/// The Rec.709-to-E-Gamut matrix with the exposure scale folded in (computed in double, rounded once).
fn composed_matrix(scale: f64) -> [[f32; 3]; 3] {
    REC709_TO_E_GAMUT.map(|row| row.map(|value| (value * scale) as f32))
}

/// A diagonal matrix holding the exposure scale.
fn diagonal_matrix(scale: f64) -> [[f32; 3]; 3] {
    let value = scale as f32;
    [[value, 0.0, 0.0], [0.0, value, 0.0], [0.0, 0.0, value]]
}

/// OCIO's `sRGB` display encode: `MONCURVE` reverse on the SSE path, with its two-rounding affine.
fn srgb_encode(rgb: [f32; 3]) -> [f32; 3] {
    rgb.map(|value| {
        let power = fast_pow(value, ENCODE_GAMMA);
        let encoded = power * ENCODE_SCALE - ENCODE_OFFSET;
        if value > ENCODE_BREAK {
            encoded
        } else {
            value * ENCODE_SLOPE
        }
    })
}

/// Display-referred view output for scene-linear `rgb` after exposure (`2^exposure`). `Filmic` returns
/// its curve output directly: OCIO's optimiser cancels the sRGB decode and encode that follow it.
/// `Raw` returns linear values. The other views end in the sRGB encode.
pub(super) fn display_view(view: ViewTransform, exposure: f64, rgb: [f32; 3]) -> Result<[f32; 3]> {
    let scale = 2.0_f64.powf(exposure);
    match view {
        ViewTransform::Standard => Ok(srgb_encode(mat3(rgb, &diagonal_matrix(scale)))),
        ViewTransform::Raw => Ok(mat3(rgb, &diagonal_matrix(scale))),
        ViewTransform::AgX => {
            let lut = loaded(LazyLock::force(&AGX_BASE_SRGB))?;
            let position = lg2_position(
                mat3(rgb, &composed_matrix(scale)),
                AGX_LOG2_MIN,
                AGX_LOG2_MAX,
            );
            let decoded = lut
                .sample(position)
                .map(|value| fast_pow(value, DECODE_EXPONENT));
            Ok(srgb_encode(decoded))
        }
        ViewTransform::Filmic => {
            let desaturation = loaded(LazyLock::force(&FILMIC_DESAT_33))?;
            let curve = loaded(LazyLock::force(&FILMIC_TO_0_70))?;
            let uniform = (1.0 / FILMIC_UNIFORM_MAX) as f32;
            let position = lg2_position(
                mat3(rgb, &diagonal_matrix(scale)),
                FILMIC_LOG2_MIN,
                FILMIC_LOG2_MAX,
            );
            let desaturated = desaturation
                .sample(position)
                .map(|value| value.mul_add(uniform, 0.0));
            Ok(desaturated.map(|value| curve.sample_channel(value, 0)))
        }
        ViewTransform::FalseColor => {
            let rec2020 = loaded(LazyLock::force(&AGX_BASE_REC2020))?;
            let palette = loaded(LazyLock::force(&AGX_FALSE_COLOR))?;
            let position = lg2_position(
                mat3(rgb, &composed_matrix(scale)),
                AGX_LOG2_MIN,
                AGX_LOG2_MAX,
            );
            let linear = rec2020
                .sample(position)
                .map(|value| fast_pow(value, DECODE_EXPONENT));
            let weight = FALSE_COLOR_LUMA.map(|value| value as f32);
            let luma = mat3(linear, &[weight, weight, weight])[0];
            let palette_position = fast_pow(luma, (1.0 / FALSE_COLOR_EXPONENT) as f32);
            let colour =
                std::array::from_fn(|channel| palette.sample_channel(palette_position, channel));
            Ok(srgb_encode(
                colour.map(|value| fast_pow(value, DECODE_EXPONENT)),
            ))
        }
    }
}

/// Cube 3D LUT with `size^3` nodes stored in file order (red fastest).
struct Lut3d {
    size: usize,
    data: Vec<[f32; 3]>,
}

impl Lut3d {
    fn node(&self, red: usize, green: usize, blue: usize) -> [f32; 3] {
        self.data[red + self.size * (green + self.size * blue)]
    }

    /// Tetrahedral interpolation with OCIO's scalar branch order. Each branch reads the four corners
    /// it blends, and the sum is accumulated with fused multiply-adds after the first product.
    fn sample(&self, rgb: [f32; 3]) -> [f32; 3] {
        let last = self.size - 1;
        let (r0, r1, fx) = lattice(rgb[0], last);
        let (g0, g1, fy) = lattice(rgb[1], last);
        let (b0, b1, fz) = lattice(rgb[2], last);
        let corner = |red: bool, green: bool, blue: bool| {
            self.node(
                if red { r1 } else { r0 },
                if green { g1 } else { g0 },
                if blue { b1 } else { b0 },
            )
        };
        if fx > fy {
            if fy > fz {
                fused_sum([
                    (1.0 - fx, corner(false, false, false)),
                    (fx - fy, corner(true, false, false)),
                    (fy - fz, corner(true, true, false)),
                    (fz, corner(true, true, true)),
                ])
            } else if fx > fz {
                fused_sum([
                    (1.0 - fx, corner(false, false, false)),
                    (fx - fz, corner(true, false, false)),
                    (fz - fy, corner(true, false, true)),
                    (fy, corner(true, true, true)),
                ])
            } else {
                fused_sum([
                    (1.0 - fz, corner(false, false, false)),
                    (fz - fx, corner(false, false, true)),
                    (fx - fy, corner(true, false, true)),
                    (fy, corner(true, true, true)),
                ])
            }
        } else if fz > fy {
            fused_sum([
                (1.0 - fz, corner(false, false, false)),
                (fz - fy, corner(false, false, true)),
                (fy - fx, corner(false, true, true)),
                (fx, corner(true, true, true)),
            ])
        } else if fz > fx {
            fused_sum([
                (1.0 - fy, corner(false, false, false)),
                (fy - fz, corner(false, true, false)),
                (fz - fx, corner(false, true, true)),
                (fx, corner(true, true, true)),
            ])
        } else {
            fused_sum([
                (1.0 - fy, corner(false, false, false)),
                (fy - fx, corner(false, true, false)),
                (fx - fz, corner(true, true, false)),
                (fz, corner(true, true, true)),
            ])
        }
    }
}

/// Sums four weighted nodes per channel: the first product is rounded, then each term is fused.
fn fused_sum(terms: [(f32, [f32; 3]); 4]) -> [f32; 3] {
    std::array::from_fn(|channel| {
        let mut sum = terms[0].0 * terms[0].1[channel];
        for &(weight, value) in &terms[1..] {
            sum = weight.mul_add(value[channel], sum);
        }
        sum
    })
}

/// Lattice position of `value` on `[0, last]`: the lower and upper node and the fraction. NaN and
/// non-positive values map to node zero, as OCIO clamps them.
fn lattice(value: f32, last: usize) -> (usize, usize, f32) {
    let last_f = last as f32;
    let scaled = value * last_f;
    let position = if scaled > 0.0 {
        scaled.min(last_f)
    } else {
        0.0
    };
    let low = position.floor();
    (low as usize, position.ceil() as usize, position - low)
}

/// 1D LUT over `[0, 1]`; single-component files are broadcast to three channels.
struct Lut1d {
    rows: Vec<[f32; 3]>,
}

impl Lut1d {
    /// Linear interpolation for one channel: `fma(fraction, high - low, low)`.
    fn sample_channel(&self, value: f32, channel: usize) -> f32 {
        let (low, high, fraction) = lattice(value, self.rows.len() - 1);
        let low_value = self.rows[low][channel];
        fraction.mul_add(self.rows[high][channel] - low_value, low_value)
    }
}

/// Maps a lazily parsed LUT load failure to an internal error. The LUTs are compiled in, so a
/// failure is a build defect rather than user input.
fn loaded<T>(lut: &'static Result<T, String>) -> Result<&'static T> {
    lut.as_ref().map_err(|message| {
        PotError::new(
            ErrorCode::InternalError,
            format!("vendored OCIO LUT failed to load: {message}"),
        )
    })
}

fn parse_numbers(tokens: &[&str]) -> std::result::Result<Vec<f32>, std::num::ParseFloatError> {
    tokens
        .iter()
        .map(|token| token.parse::<f64>().map(|value| value as f32))
        .collect()
}

fn parse_triple(values: &[f32]) -> [f32; 3] {
    [values[0], values[1], values[2]]
}

/// Parses a `.cube` 3D LUT: the `LUT_3D_SIZE N` header and exactly `N^3` rows of three numbers.
/// Other lines (`TITLE`, `DOMAIN_*`, comments) are skipped; the vendored files use the `[0, 1]` domain.
fn parse_cube(source: &str) -> Result<Lut3d, String> {
    let mut size = None;
    let mut data = Vec::new();
    for line in source.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if let ["LUT_3D_SIZE", value] = tokens.as_slice() {
            size = Some(value.parse::<usize>().map_err(|error| error.to_string())?);
            continue;
        }
        match parse_numbers(&tokens) {
            Ok(values) if values.len() == 3 => data.push(parse_triple(&values)),
            _ => {}
        }
    }
    let size = size.ok_or_else(|| "missing LUT_3D_SIZE".to_owned())?;
    if size < 2 || data.len() != size.pow(3) {
        return Err(format!(
            "expected {} LUT rows, found {}",
            size.pow(3),
            data.len()
        ));
    }
    Ok(Lut3d { size, data })
}

/// Parses a `.spi1d` 1D LUT: the `Length N` header and exactly `N` rows of one or three numbers.
/// Other lines (`Version`, `From`, `Components`, braces) are skipped.
fn parse_spi1d(source: &str) -> Result<Lut1d, String> {
    let mut length = None;
    let mut rows = Vec::new();
    for line in source.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if let ["Length", value] = tokens.as_slice() {
            length = Some(value.parse::<usize>().map_err(|error| error.to_string())?);
            continue;
        }
        match parse_numbers(&tokens) {
            Ok(values) if values.len() == 1 => rows.push([values[0]; 3]),
            Ok(values) if values.len() == 3 => rows.push(parse_triple(&values)),
            _ => {}
        }
    }
    let length = length.ok_or_else(|| "missing Length header".to_owned())?;
    if length < 2 || rows.len() != length {
        return Err(format!("expected {length} rows, found {}", rows.len()));
    }
    Ok(Lut1d { rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tetrahedral_sampling_returns_node_values_at_lattice_points() {
        let size = 3;
        let data: Vec<[f32; 3]> = (0..size)
            .flat_map(|blue| {
                (0..size).flat_map(move |green| {
                    (0..size)
                        .map(move |red| [red as f32 / 2.0, green as f32 / 2.0, blue as f32 / 2.0])
                })
            })
            .collect();
        let lut = Lut3d { size, data };
        for blue in 0..size {
            for green in 0..size {
                for red in 0..size {
                    let expected = [red as f32 / 2.0, green as f32 / 2.0, blue as f32 / 2.0];
                    let sampled = lut.sample(expected);
                    for channel in 0..3 {
                        assert!((sampled[channel] - expected[channel]).abs() < 1.0e-6);
                    }
                }
            }
        }
    }

    #[test]
    fn composed_matrix_rows_sum_to_one() {
        for row in composed_matrix(1.0) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1.0e-6, "row sums to {sum}");
        }
    }

    #[test]
    fn fast_pow_is_close_to_libm_for_positive_bases() {
        for value in [0.001_f32, 0.18, 1.0, 4.5] {
            let fast = f64::from(fast_pow(value, DECODE_EXPONENT));
            let exact = f64::from(value).powf(f64::from(DECODE_EXPONENT));
            assert!(
                (fast - exact).abs() < 1.0e-4 * exact.max(1.0),
                "{value}: {fast} vs {exact}"
            );
        }
    }

    #[test]
    fn vendored_luts_parse_with_expected_entry_counts() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(parse_cube(AGX_BASE_SRGB_CUBE)?.data.len(), 57 * 57 * 57);
        assert_eq!(parse_cube(AGX_BASE_REC2020_CUBE)?.data.len(), 57 * 57 * 57);
        assert_eq!(parse_cube(FILMIC_DESAT_33_CUBE)?.data.len(), 33 * 33 * 33);
        assert_eq!(parse_spi1d(AGX_FALSE_COLOR_SPI1D)?.rows.len(), 4096);
        assert_eq!(parse_spi1d(FILMIC_TO_0_70_SPI1D)?.rows.len(), 4096);
        Ok(())
    }
}
