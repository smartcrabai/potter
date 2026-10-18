//! Blender 5.2.2 `OpenColorIO` display views for the `sRGB` display, with Blender's looks, in float32.
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
//! - Exposure is folded into the first matrix in double precision, then rounded to float32 once.
//!
//! A look runs as Blender's display processor builds it: the look's `LookTransform` into its process
//! space, the look operation, then the `DisplayViewTransform` back to scene light and into the view.
//! `OpenColorIO`'s optimiser then merges adjacent affine operations in double precision, and cancels a
//! `log2`/`exp2` pair only where the operations between them are exactly the identity. The look paths
//! below are the optimised sequences; `color/looks.rs` names the looks.
//!
//! The FMA choice is the one measured against Blender's default processor on macOS arm64. Linux
//! `x86_64` may round some of these operations differently; the bounds in the Blender oracle tests
//! cover that (see `tests/color_view_transform_blender.rs` and `tests/color_looks_blender.rs`).
//!
//! The LUTs in `color/luts/` are byte-for-byte copies of Blender 5.2.2 data. Source, SHA-256,
//! authors, and licence status are recorded in `color/luts/NOTICE`.

use std::sync::LazyLock;

use super::ViewTransform;
use super::looks::{AgxOp, FilmicCurve, Look, LookOps};
use crate::error::{ErrorCode, PotError, Result};

const AGX_BASE_SRGB_CUBE: &str = include_str!("luts/AgX_Base_sRGB.cube");
const AGX_BASE_REC2020_CUBE: &str = include_str!("luts/AgX_Base_Rec2020.cube");
const AGX_FALSE_COLOR_SPI1D: &str = include_str!("luts/AgX_False_Color.spi1d");
const FILMIC_DESAT_33_CUBE: &str = include_str!("luts/filmic_desat_33.cube");
const FILMIC_TO_0_70_SPI1D: &str = include_str!("luts/filmic_to_0-70_1-03.spi1d");
const FILMIC_TO_1_20_SPI1D: &str = include_str!("luts/filmic_to_1.20_1-00.spi1d");
const FILMIC_TO_0_99_SPI1D: &str = include_str!("luts/filmic_to_0.99_1-0075.spi1d");
const FILMIC_TO_0_85_SPI1D: &str = include_str!("luts/filmic_to_0-85_1-011.spi1d");
const FILMIC_TO_0_60_SPI1D: &str = include_str!("luts/filmic_to_0-60_1-04.spi1d");
const FILMIC_TO_0_48_SPI1D: &str = include_str!("luts/filmic_to_0-48_1-09.spi1d");
const FILMIC_TO_0_35_SPI1D: &str = include_str!("luts/filmic_to_0-35_1-30.spi1d");
const LUMINANCE_COMPENSATION_CUBE: &str = include_str!("luts/luminance_compensation_bt2020.cube");

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
static LUMINANCE_COMPENSATION: LazyLock<Result<Lut3d, String>> =
    LazyLock::new(|| parse_cube(LUMINANCE_COMPENSATION_CUBE));

static FILMIC_VERY_HIGH: LazyLock<Result<Vec<f32>, String>> =
    LazyLock::new(|| compose_filmic_look(FILMIC_TO_1_20_SPI1D));
static FILMIC_HIGH: LazyLock<Result<Vec<f32>, String>> =
    LazyLock::new(|| compose_filmic_look(FILMIC_TO_0_99_SPI1D));
static FILMIC_MEDIUM_HIGH: LazyLock<Result<Vec<f32>, String>> =
    LazyLock::new(|| compose_filmic_look(FILMIC_TO_0_85_SPI1D));
static FILMIC_MEDIUM_LOW: LazyLock<Result<Vec<f32>, String>> =
    LazyLock::new(|| compose_filmic_look(FILMIC_TO_0_60_SPI1D));
static FILMIC_LOW: LazyLock<Result<Vec<f32>, String>> =
    LazyLock::new(|| compose_filmic_look(FILMIC_TO_0_48_SPI1D));
static FILMIC_VERY_LOW: LazyLock<Result<Vec<f32>, String>> =
    LazyLock::new(|| compose_filmic_look(FILMIC_TO_0_35_SPI1D));

/// `lg2` allocation window of the `AgX` views and of `AgX Log` (`vars` in the config, float32).
const AGX_LOG2_MIN: f64 = -12.47393;
const AGX_LOG2_MAX: f64 = 12.526_068_811_7;
/// `lg2` allocation window of `Filmic Log` and the Filmic view, and its to-scene upper end.
const FILMIC_LOG2_MIN: f64 = -12.473_931_188;
const FILMIC_LOG2_MAX: f64 = 12.526_068_812;
const FILMIC_TO_SCENE_MAX: f64 = 4.026_068_812;
/// `uniform` allocation upper bound of the `Filmic Log` desaturation output.
const FILMIC_UNIFORM_MAX: f64 = 0.66;

const FALSE_COLOR_EXPONENT: f64 = 2.5;

/// Rec.709 linear to `FilmLight` E-Gamut linear, `E * inverse(Rec.709)` in the XYZ D65 frame.
const REC709_TO_E_GAMUT: [[f64; 3]; 3] = [
    [0.5593711333100773, 0.30478334662300893, 0.13584556048047705],
    [0.07622070548350629, 0.7879717701028465, 0.13580746617472153],
    [0.06552671177877019, 0.16454675460372414, 0.7699265016503745],
];

/// Rec.2020 (`lum-comp` output) to the AgX inset, `MI * R2020 * E^-1`: the forward `AgX Log` tail.
const AGX_LOG_TAIL: [[f64; 3]; 3] = [
    [
        0.9679715035369568,
        0.11789008541573688,
        -0.08586162568910853,
    ],
    [
        0.13181009198676008,
        0.9160783765112859,
        -0.04788856416567144,
    ],
    [
        0.04687865295063367,
        0.012432801782063642,
        0.9406886085162658,
    ],
];

/// The AgX inset inverse and `E * R2020^-1` of the `AgX` view: `DisplayViewTransform` to scene light.
const AGX_VIEW_FROM_LOG: [[f64; 3]; 3] = [
    [
        1.0473107847120087,
        -0.13598180076490982,
        0.08867103590993768,
    ],
    [-0.1533149367932694, 1.1107622300653885, 0.04255280466820433],
    [
        -0.050165787234335076,
        -0.007904042762326017,
        1.0580697604757736,
    ],
];

/// `AgX_Log_TAIL` followed by `AGX_VIEW_FROM_LOG` (identity to within rounding): `Base Contrast`.
const AGX_BASE_LOOK: [[f64; 3]; 3] = [
    [
        1.0000000000000002,
        -8.92176963811131e-17,
        1.532196043848371e-17,
    ],
    [
        -1.5752959025183133e-17,
        1.0000000000000004,
        -4.139178090097987e-17,
    ],
    [
        8.430357809639775e-18,
        7.16965736967217e-18,
        0.9999999999999999,
    ],
];

/// `Greyscale`: the luma matrix between the two `AgX` stages, with the stages' matrices folded in.
const AGX_GREYSCALE_LOOK: [[f64; 3]; 3] = [
    [0.3372217961815739, 0.5914124854316711, 0.07136567858584152],
    [0.3372218225129597, 0.591412531611101, 0.07136568415830817],
    [0.3372217660413904, 0.591412432572452, 0.07136567220732544],
];

/// Luma weights of the `AgX False Color Rec.709` view, applied to Rec.2020 linear values.
const FALSE_COLOR_LUMA: [f64; 3] = [
    0.258_923_535_568_984_8,
    0.610_498_534_606_652_5,
    0.130_577_929_824_362_84,
];

/// Exponent of the Rec.1886 decode (`AgX`, `FalseColor`) and of the gamma stage's inverse pair.
const DECODE_EXPONENT: f32 = 2.4;

/// `FLT_MIN` clamp before `log2` (`LogRendererSSE`).
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

/// Entries of the composed `Filmic Log` look table: a uniform table over `[0, 1]`.
const COMPOSED_LUT_SIZE: usize = 65536;

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

/// A `lg2` allocation bound as the config stores it: a float32 `vars` entry, widened to double.
fn window(bound: f64) -> f64 {
    f64::from(bound as f32)
}

/// Forward `lg2` fit (`MatrixTransform::Fit`): slope `1/(max - min)` and offset `-min/(max - min)`,
/// computed in double from the window and rounded to float32, applied as `fma(value, slope, offset)`.
fn fit_lg2(value: f32, old_min: f64, old_max: f64) -> f32 {
    let (low, high) = (window(old_min), window(old_max));
    let denominator = high - low;
    value.mul_add((1.0 / denominator) as f32, (-low / denominator) as f32)
}

/// `lg2` allocation forward (`LogRendererSSE`, then the fit), per channel.
fn lg2_position(rgb: [f32; 3], old_min: f64, old_max: f64) -> [f32; 3] {
    rgb.map(|value| fit_lg2(fast_log2(value.max(FLT_MIN)), old_min, old_max))
}

/// Inverse `lg2` allocation: the inverse fit (slope `max - min`, offset `min`) then `AntiLog2`.
fn inverse_exp2(rgb: [f32; 3], old_min: f64, old_max: f64) -> [f32; 3] {
    let (low, high) = (window(old_min), window(old_max));
    let slope = (high - low) as f32;
    let offset = low as f32;
    rgb.map(|value| fast_exp2(value.mul_add(slope, offset)))
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

/// A float64 matrix rounded to float32 once per entry, as OCIO's renderers store it.
fn to_f32(matrix: &[[f64; 3]; 3]) -> [[f32; 3]; 3] {
    matrix.map(|row| row.map(|value| value as f32))
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

/// Display-referred view output for scene-linear `rgb` after exposure (`2^exposure`), with `look`.
/// Returns the view output before gamma. `Filmic` returns its curve output directly: OCIO's optimiser
/// cancels the sRGB decode and encode that follow it. `Raw` returns linear values. The other views end
/// in the sRGB encode.
///
/// Errors with [`ErrorCode::InvalidArgument`] when `look` is not listed for `view`, and with
/// [`ErrorCode::InternalError`] if a vendored LUT fails to load.
pub fn display_view(
    view: ViewTransform,
    look: Look,
    exposure: f64,
    rgb: [f32; 3],
) -> Result<[f32; 3]> {
    if !look.is_compatible_with(view) {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            format!("color look {look:?} does not apply to view {view:?}"),
        ));
    }
    let scale = 2.0_f64.powf(exposure);
    match look.ops() {
        LookOps::None => display_without_look(view, scale, rgb),
        LookOps::Filmic(curve) => filmic_look_view(view, curve, scale, rgb),
        LookOps::Agx(op) => agx_look_view(view, op, scale, rgb),
    }
}

fn display_without_look(view: ViewTransform, scale: f64, rgb: [f32; 3]) -> Result<[f32; 3]> {
    match view {
        ViewTransform::Standard => Ok(srgb_encode(mat3(rgb, &diagonal_matrix(scale)))),
        ViewTransform::Raw => Ok(mat3(rgb, &diagonal_matrix(scale))),
        ViewTransform::Filmic => filmic_view_tail(lg2_position(
            mat3(rgb, &diagonal_matrix(scale)),
            FILMIC_LOG2_MIN,
            FILMIC_LOG2_MAX,
        )),
        ViewTransform::AgX | ViewTransform::FalseColor => agx_view_from_position(
            view,
            lg2_position(
                mat3(rgb, &composed_matrix(scale)),
                AGX_LOG2_MIN,
                AGX_LOG2_MAX,
            ),
        ),
    }
}

/// The `Filmic sRGB` view after the `Filmic Log` from-scene `lg2` position: the desaturation LUT, the
/// uniform fit, then `filmic_to_0-70`.
fn filmic_view_tail(position: [f32; 3]) -> Result<[f32; 3]> {
    let desaturation = loaded(LazyLock::force(&FILMIC_DESAT_33))?;
    let curve = loaded(LazyLock::force(&FILMIC_TO_0_70))?;
    let uniform = filmic_uniform_scale() as f32;
    let desaturated = desaturation
        .sample(position)
        .map(|value| value.mul_add(uniform, 0.0));
    Ok(desaturated.map(|value| curve.sample_channel(value, 0)))
}

/// The `Filmic Log` uniform fit `[0, 0.66] -> [0, 1]`: `1 / bound` in double, from the float32 bound.
fn filmic_uniform_scale() -> f64 {
    1.0 / window(FILMIC_UNIFORM_MAX)
}

/// `Filmic` (`Filmic sRGB`), `Standard` and `Raw` with a `Filmic Log` look.
///
/// The look's `Filmic Log` from-scene (diagonal, `lg2`, desaturation LUT, uniform fit) gives `d`. A
/// curve look maps `d * uniform` through the composed 65536-entry table. OCIO then runs the to-scene
/// inverse `lg2` (window `[-12.47, 4.03]`) and the view's own `lg2`, whose exp2 and log2 cancel against
/// each other; `Filmic` keeps the combined scale of that pair, `0.66`, and `Standard` and `Raw` keep the
/// inverse `lg2` alone. `Medium Contrast` has no curve, so its uniform fit folds into the same scale.
fn filmic_look_view(
    view: ViewTransform,
    curve: Option<FilmicCurve>,
    scale: f64,
    rgb: [f32; 3],
) -> Result<[f32; 3]> {
    let desaturation = loaded(LazyLock::force(&FILMIC_DESAT_33))?;
    let d = desaturation.sample(lg2_position(
        mat3(rgb, &diagonal_matrix(scale)),
        FILMIC_LOG2_MIN,
        FILMIC_LOG2_MAX,
    ));
    let uniform = filmic_uniform_scale();
    let view_scale = 1.0 / (window(FILMIC_LOG2_MAX) - window(FILMIC_LOG2_MIN));
    let to_scene_scale = window(FILMIC_TO_SCENE_MAX) - window(FILMIC_LOG2_MIN);
    match curve {
        Some(curve) => {
            let table = filmic_look_table(curve)?;
            let looked = d.map(|value| lerp_uniform(table, value.mul_add(uniform as f32, 0.0)));
            match view {
                ViewTransform::Filmic => {
                    let boundary = (view_scale * to_scene_scale) as f32;
                    filmic_view_tail(looked.map(|value| value * boundary))
                }
                ViewTransform::Standard => Ok(srgb_encode(inverse_exp2(
                    looked,
                    FILMIC_LOG2_MIN,
                    FILMIC_TO_SCENE_MAX,
                ))),
                _ => Ok(inverse_exp2(looked, FILMIC_LOG2_MIN, FILMIC_TO_SCENE_MAX)),
            }
        }
        None => match view {
            ViewTransform::Filmic => {
                let boundary = (view_scale * (to_scene_scale * uniform)) as f32;
                filmic_view_tail(d.map(|value| value * boundary))
            }
            _ => {
                let slope = (to_scene_scale * uniform) as f32;
                let offset = window(FILMIC_LOG2_MIN) as f32;
                let linear = d.map(|value| fast_exp2(value.mul_add(slope, offset)));
                Ok(if view == ViewTransform::Standard {
                    srgb_encode(linear)
                } else {
                    linear
                })
            }
        },
    }
}

/// Linear interpolation on a uniform table over `[0, 1]`: `fma(fraction, high - low, low)`.
fn lerp_uniform(table: &[f32], value: f32) -> f32 {
    let (low, high, fraction) = lattice(value, table.len() - 1);
    let low_value = table[low];
    fraction.mul_add(table[high] - low_value, low_value)
}

/// The composed `Filmic Log` look table for `curve`, cached on first use.
fn filmic_look_table(curve: FilmicCurve) -> Result<&'static [f32]> {
    let table = match curve {
        FilmicCurve::VeryHigh => &FILMIC_VERY_HIGH,
        FilmicCurve::High => &FILMIC_HIGH,
        FilmicCurve::MediumHigh => &FILMIC_MEDIUM_HIGH,
        FilmicCurve::MediumLow => &FILMIC_MEDIUM_LOW,
        FilmicCurve::Low => &FILMIC_LOW,
        FilmicCurve::VeryLow => &FILMIC_VERY_LOW,
    };
    loaded(LazyLock::force(table)).map(Vec::as_slice)
}

/// Composes a `filmic_to_*` look curve with the inverse of `filmic_to_0-70` into one uniform table
/// (`COMPOSE_RESAMPLE_BIG`): `y_i = B^-1(A(x_i))`, with `x_i = i / 65535`.
fn compose_filmic_look(look_source: &str) -> std::result::Result<Vec<f32>, String> {
    let look = parse_spi1d(look_source)?;
    let curve = LazyLock::force(&FILMIC_TO_0_70)
        .as_ref()
        .map_err(String::clone)?;
    let inverse = monotone(curve.rows.iter().map(|row| row[0]).collect());
    let step = 1.0 / (COMPOSED_LUT_SIZE - 1) as f32;
    Ok((0..COMPOSED_LUT_SIZE)
        .map(|index| {
            let x = index as f32 * step;
            find_lut_inv(&inverse, look.sample_channel(x, 0))
        })
        .collect())
}

/// Makes a table non-decreasing: a value not above its predecessor takes the predecessor's value.
fn monotone(mut values: Vec<f32>) -> Vec<f32> {
    let mut previous = f32::NEG_INFINITY;
    for value in &mut values {
        if *value <= previous {
            *value = previous;
        }
        previous = *value;
    }
    values
}

/// `OpenColorIO`'s `FindLutInv` on a monotone table over `[0, 1]`: the position of `value` in the
/// table, as a fraction of the last index, by linear interpolation between the two bracketing entries.
fn find_lut_inv(table: &[f32], value: f32) -> f32 {
    let last = table.len() - 1;
    let clamped = value.max(table[0]).min(table[last]);
    let upper = table[..last].partition_point(|&entry| entry < clamped);
    let lower = upper.saturating_sub(1);
    let next = if lower < last { lower + 1 } else { lower };
    let delta = if table[next] > table[lower] {
        (clamped - table[lower]) / (table[next] - table[lower])
    } else {
        0.0
    };
    (lower as f32 + delta) * (1.0 / last as f32)
}

/// `AgX` (`AgX Base Rec.1886`) and `False Color` (`AgX False Color Rec.709`) with an `AgX Log` look.
///
/// The look's from-scene (`E-Gamut` matrix, `lg2`, the luminance compensation LUT, its inverse `lg2`,
/// and the inset) gives the linear value `lin`. The look operation then acts on the `lg2` position. The
/// to-scene and view stages after it fold into the matrix `AGX_VIEW_FROM_LOG`, or into the `Greyscale`
/// matrix, or into the identity `Base Contrast` (where the log2/exp2 pair around the identity stays).
fn agx_look_view(view: ViewTransform, op: AgxOp, scale: f64, rgb: [f32; 3]) -> Result<[f32; 3]> {
    let compensation = loaded(LazyLock::force(&LUMINANCE_COMPENSATION))?;
    let position = lg2_position(
        mat3(rgb, &composed_matrix(scale)),
        AGX_LOG2_MIN,
        AGX_LOG2_MAX,
    );
    let lin = inverse_exp2(compensation.sample(position), AGX_LOG2_MIN, AGX_LOG2_MAX);
    let position = match op {
        AgxOp::Base => lg2_position(
            mat3(lin, &to_f32(&AGX_BASE_LOOK)),
            AGX_LOG2_MIN,
            AGX_LOG2_MAX,
        ),
        AgxOp::Greyscale => lg2_position(
            mat3(lin, &to_f32(&AGX_GREYSCALE_LOOK)),
            AGX_LOG2_MIN,
            AGX_LOG2_MAX,
        ),
        AgxOp::Punchy => {
            let log = lg2_position(
                mat3(lin, &to_f32(&AGX_LOG_TAIL)),
                AGX_LOG2_MIN,
                AGX_LOG2_MAX,
            );
            let graded = cdl_power(grading_tone_shadows(log), 1.0912);
            lg2_position(
                mat3(
                    inverse_exp2(graded, AGX_LOG2_MIN, AGX_LOG2_MAX),
                    &to_f32(&AGX_VIEW_FROM_LOG),
                ),
                AGX_LOG2_MIN,
                AGX_LOG2_MAX,
            )
        }
        AgxOp::Grade {
            contrast,
            saturation,
        } => {
            let log = lg2_position(
                mat3(lin, &to_f32(&AGX_LOG_TAIL)),
                AGX_LOG2_MIN,
                AGX_LOG2_MAX,
            );
            let graded = grading_primary(log, contrast, saturation);
            lg2_position(
                mat3(
                    inverse_exp2(graded, AGX_LOG2_MIN, AGX_LOG2_MAX),
                    &to_f32(&AGX_VIEW_FROM_LOG),
                ),
                AGX_LOG2_MIN,
                AGX_LOG2_MAX,
            )
        }
    };
    agx_view_from_position(view, position)
}

/// The `AgX` or `False Color` view from its `lg2` position (the view's from-scene ends here).
fn agx_view_from_position(view: ViewTransform, position: [f32; 3]) -> Result<[f32; 3]> {
    match view {
        ViewTransform::AgX => {
            let lut = loaded(LazyLock::force(&AGX_BASE_SRGB))?;
            let decoded = lut
                .sample(position)
                .map(|value| fast_pow(value, DECODE_EXPONENT));
            Ok(srgb_encode(decoded))
        }
        _ => {
            let rec2020 = loaded(LazyLock::force(&AGX_BASE_REC2020))?;
            let palette = loaded(LazyLock::force(&AGX_FALSE_COLOR))?;
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

/// `CDL` power, forward, no clamp: negative values pass through, `NaN` becomes zero, then the
/// saturation pass runs unconditionally at saturation 1 (`CDLOpCPU.cpp`).
fn cdl_power(rgb: [f32; 3], power: f32) -> [f32; 3] {
    let powered = rgb.map(|value| {
        if value.is_nan() {
            0.0
        } else if value < 0.0 {
            value
        } else {
            fast_pow(value, power)
        }
    });
    saturate(powered, 1.0)
}

/// Saturation with the Rec.709 weights: `luma + sat * (value - luma)`, with `luma` in the order
/// `(t0 + t1) + t2` (`GradingPrimary` and `CDL`, scalar form, fused).
fn saturate(rgb: [f32; 3], saturation: f32) -> [f32; 3] {
    const WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];
    let t0 = rgb[0] * WEIGHTS[0];
    let t1 = rgb[1] * WEIGHTS[1];
    let t2 = rgb[2] * WEIGHTS[2];
    let luma = (t0 + t1) + t2;
    rgb.map(|value| saturation.mul_add(value - luma, luma))
}

/// `GradingPrimary` in log style for an `AgX` grade: contrast about the pivot 0.4, then saturation
/// when it is not 1 (`GradingPrimaryOpCPU.cpp`, offset and gamma are identities here).
fn grading_primary(rgb: [f32; 3], contrast: f32, saturation: Option<f32>) -> [f32; 3] {
    const PIVOT: f32 = 0.4;
    let contrasted = rgb.map(|value| (value - PIVOT).mul_add(contrast, PIVOT));
    match saturation {
        Some(saturation) => saturate(contrasted, saturation),
        None => contrasted,
    }
}

/// `GradingTone` shadows (`GradingToneOpCPU.cpp`): the R, G and B curves with rgb 0.2 and then the
/// master curve with 0.35, each on the lanes it acts on, then the `65504` clamp.
fn grading_tone_shadows(rgb: [f32; 3]) -> [f32; 3] {
    const X0: f32 = 0.1;
    const X1: f32 = 0.25;
    const X2: f32 = 0.4;
    const M2: f32 = 1.0;
    let rgb_y1 = shadow_y1(0.2);
    let master_y1 = shadow_y1(0.35);
    let channels = rgb.map(|value| highlight_shadow(value, X0, X1, X2, rgb_y1, 0.2, M2));
    let master = channels.map(|value| highlight_shadow(value, X0, X1, X2, master_y1, 0.35, M2));
    master.map(|value| value.min(65504.0))
}

/// The `y1` control value of one shadow curve (`highlightShadow_precompute`), float32.
fn shadow_y1(m0: f32) -> f32 {
    const X0: f32 = 0.1;
    const X2: f32 = 0.4;
    const M2: f32 = 1.0;
    let x1 = X0 + (X2 - X0) * 0.5;
    let left = 2.0 * X0 + m0 * (x1 - X0);
    let right = 2.0 * X2 - M2 * (X2 - x1);
    (0.5 / (X2 - X0)) * (left * (X2 - x1) + right * (x1 - X0))
}

/// `ComputeHSFwd` for one value: two quadratic segments joined at `x1`, with linear tails.
fn highlight_shadow(t: f32, x0: f32, x1: f32, x2: f32, y1: f32, m0: f32, m2: f32) -> f32 {
    let y0 = x0;
    let y2 = x2;
    let one = 1.0_f32;
    let tl = (t - x0) / (x1 - x0);
    let tr = (t - x1) / (x2 - x1);
    let left = y0 * (one - tl * tl) + y1 * tl * tl + m0 * (one - tl) * tl * (x1 - x0);
    let right =
        y1 * (one - tr) * (one - tr) + y2 * (2.0 - tr) * tr + m2 * (tr - one) * tr * (x2 - x1);
    let mut result = if t < x1 { left } else { right };
    result = if t < x0 { (t - x0) * m0 + y0 } else { result };
    if t < x2 { result } else { (t - x2) * m2 + y2 }
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

    /// One Blender 5.2.2 probe row: the view, the look token, the setting, and Blender's display values
    /// (clamped to `[0, 1]`) for the four `SAMPLES`. Probed by the chain of `tests/color_looks_blender.rs`.
    struct Probed {
        view: ViewTransform,
        look: &'static str,
        exposure: f64,
        gamma: f64,
        expected: [[f64; 3]; 4],
    }

    const fn probed(
        view: ViewTransform,
        look: &'static str,
        exposure: f64,
        gamma: f64,
        expected: [[f64; 3]; 4],
    ) -> Probed {
        Probed {
            view,
            look,
            exposure,
            gamma,
            expected,
        }
    }

    const SAMPLES: [[f32; 3]; 4] = [
        [0.05, 0.05, 0.05],
        [0.18, 0.18, 0.18],
        [0.8, 0.1, 0.05],
        [0.02, 0.5, 4.0],
    ];

    /// Float bound per view, from the fused-versus-unfused spread and the Blender residual of the
    /// emulation (see `color/looks.rs`): Filmic 1e-6, AgX 2e-5, False Color 4e-3 (palette steps).
    fn probed_tolerance(view: ViewTransform) -> f64 {
        match view {
            ViewTransform::FalseColor => 4.0e-3,
            ViewTransform::AgX => 2.0e-5,
            ViewTransform::Standard | ViewTransform::Filmic | ViewTransform::Raw => 1.0e-6,
        }
    }

    /// Blender 5.2.2 probe values (OCIO 2.5.0, default CPU processor, per pixel) for every primary-view
    /// look at both settings, clamped to `[0, 1]`.
    const PROBED: &[Probed] = &[
        probed(
            ViewTransform::Filmic,
            "very_high_contrast",
            0.0,
            1.0,
            [
                [0.176529422, 0.176529422, 0.176529422],
                [0.500329196, 0.500329196, 0.500329196],
                [0.859800994, 0.330372781, 0.176529422],
                [0.0664667711, 0.775205195, 0.980590463],
            ],
        ),
        probed(
            ViewTransform::Filmic,
            "very_high_contrast",
            0.75,
            2.0,
            [
                [0.534778178, 0.534778178, 0.534778178],
                [0.80761838, 0.80761838, 0.80761838],
                [0.959898531, 0.692724764, 0.534778178],
                [0.353749394, 0.931211472, 0.995766997],
            ],
        ),
        probed(
            ViewTransform::Filmic,
            "high_contrast",
            0.0,
            1.0,
            [
                [0.214304075, 0.214304075, 0.214304075],
                [0.500280321, 0.500280321, 0.500280321],
                [0.830248773, 0.354545861, 0.214304075],
                [0.0979886651, 0.74499476, 0.974515319],
            ],
        ),
        probed(
            ViewTransform::Filmic,
            "high_contrast",
            0.75,
            2.0,
            [
                [0.561659455, 0.561659455, 0.561659455],
                [0.794838071, 0.794838071, 0.794838071],
                [0.948177576, 0.694860458, 0.561659455],
                [0.397795767, 0.915453553, 0.994820654],
            ],
        ),
        probed(
            ViewTransform::Filmic,
            "medium_high_contrast",
            0.0,
            1.0,
            [
                [0.243725345, 0.243725345, 0.243725345],
                [0.50024426, 0.50024426, 0.50024426],
                [0.800742269, 0.372163743, 0.243725345],
                [0.126407132, 0.718648553, 0.961226881],
            ],
        ),
        probed(
            ViewTransform::Filmic,
            "medium_high_contrast",
            0.75,
            2.0,
            [
                [0.580663025, 0.580663025, 0.580663025],
                [0.784662724, 0.784662724, 0.784662724],
                [0.933865547, 0.696412146, 0.580663025],
                [0.430800617, 0.899162412, 0.991197765],
            ],
        ),
        probed(
            ViewTransform::Filmic,
            "medium_contrast",
            0.0,
            1.0,
            [
                [0.273077697, 0.273077697, 0.273077697],
                [0.500211298, 0.500211298, 0.500211298],
                [0.773816407, 0.38828516, 0.273077697],
                [0.159214944, 0.694953322, 0.949384511],
            ],
        ),
        probed(
            ViewTransform::Filmic,
            "medium_contrast",
            0.75,
            2.0,
            [
                [0.597834349, 0.597834349, 0.597834349],
                [0.775460005, 0.775460005, 0.775460005],
                [0.920427799, 0.697772026, 0.597834349],
                [0.46718657, 0.884039819, 0.98821044],
            ],
        ),
        probed(
            ViewTransform::Filmic,
            "medium_low_contrast",
            0.0,
            1.0,
            [
                [0.29491204, 0.29491204, 0.29491204],
                [0.500185907, 0.500185907, 0.500185907],
                [0.748369217, 0.400060803, 0.29491204],
                [0.185329393, 0.674463689, 0.931670308],
            ],
        ),
        probed(
            ViewTransform::Filmic,
            "medium_low_contrast",
            0.75,
            2.0,
            [
                [0.610084355, 0.610084355, 0.610084355],
                [0.767964363, 0.767964363, 0.767964363],
                [0.905863285, 0.698777735, 0.610084355],
                [0.493122697, 0.869350076, 0.982635617],
            ],
        ),
        probed(
            ViewTransform::Filmic,
            "low_contrast",
            0.0,
            1.0,
            [
                [0.316261113, 0.316261113, 0.316261113],
                [0.500160336, 0.500160336, 0.500160336],
                [0.723724127, 0.410910964, 0.316261113],
                [0.213421956, 0.654702723, 0.914260745],
            ],
        ),
        probed(
            ViewTransform::Filmic,
            "low_contrast",
            0.75,
            2.0,
            [
                [0.621308148, 0.621308148, 0.621308148],
                [0.760641932, 0.760641932, 0.760641932],
                [0.89140898, 0.699671268, 0.621308148],
                [0.518178225, 0.854873538, 0.977207303],
            ],
        ),
        probed(
            ViewTransform::Filmic,
            "very_low_contrast",
            0.0,
            1.0,
            [
                [0.33369869, 0.33369869, 0.33369869],
                [0.50013572, 0.50013572, 0.50013572],
                [0.700187147, 0.419320315, 0.33369869],
                [0.23832497, 0.636083901, 0.896939814],
            ],
        ),
        probed(
            ViewTransform::Filmic,
            "very_low_contrast",
            0.75,
            2.0,
            [
                [0.629975915, 0.629975915, 0.629975915],
                [0.753727734, 0.753727734, 0.753727734],
                [0.877118945, 0.700339079, 0.629975915],
                [0.538441718, 0.840792656, 0.97186774],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_punchy",
            0.0,
            1.0,
            [
                [0.136080414, 0.136079669, 0.136095315],
                [0.346289992, 0.346287489, 0.346319467],
                [0.685787678, 0.263768435, 0.150582224],
                [0.470408618, 0.636550188, 0.917137027],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_punchy",
            0.75,
            2.0,
            [
                [0.459255099, 0.459253967, 0.459277689],
                [0.673999906, 0.673998713, 0.674025297],
                [0.891004682, 0.61070168, 0.520409465],
                [0.771240115, 0.850264668, 0.979257464],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_greyscale",
            0.0,
            1.0,
            [
                [0.227206677, 0.227205575, 0.227228582],
                [0.461317956, 0.461316228, 0.461352646],
                [0.538577855, 0.538576245, 0.538610995],
                [0.773627639, 0.773626268, 0.773648798],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_greyscale",
            0.75,
            2.0,
            [
                [0.555030465, 0.555029154, 0.555055737],
                [0.756242633, 0.756241679, 0.756263971],
                [0.800086498, 0.800085664, 0.80010426],
                [0.914033771, 0.914033353, 0.91404295],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_very_high_contrast",
            0.0,
            1.0,
            [
                [0.141966283, 0.141965568, 0.141981602],
                [0.461314201, 0.461312294, 0.461348832],
                [0.902942717, 0.337300301, 0.189112544],
                [0.715783179, 0.824373066, 1.0],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_very_high_contrast",
            0.75,
            2.0,
            [
                [0.489413202, 0.489412248, 0.489436835],
                [0.793152094, 0.793151259, 0.793170452],
                [0.99497056, 0.732552767, 0.664865255],
                [0.90849638, 0.950027704, 1.0],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_high_contrast",
            0.0,
            1.0,
            [
                [0.164576679, 0.164575905, 0.164593905],
                [0.461315095, 0.461313188, 0.461349308],
                [0.881120086, 0.341692775, 0.198981047],
                [0.6739012, 0.794572055, 1.0],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_high_contrast",
            0.75,
            2.0,
            [
                [0.507988214, 0.507986784, 0.508012056],
                [0.782423019, 0.782422185, 0.782442331],
                [0.983291745, 0.71986413, 0.647107899],
                [0.885970235, 0.934144497, 1.0],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_medium_high_contrast",
            0.0,
            1.0,
            [
                [0.19403252, 0.194031686, 0.194051981],
                [0.461315989, 0.461313784, 0.461350024],
                [0.847137272, 0.356407762, 0.231058538],
                [0.628358722, 0.758717, 0.992565453],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_medium_high_contrast",
            0.75,
            2.0,
            [
                [0.531888902, 0.53188771, 0.531913221],
                [0.769447803, 0.769446731, 0.769467831],
                [0.963584423, 0.710349262, 0.638836265],
                [0.858267367, 0.913585603, 1.0],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_base_contrast",
            0.0,
            1.0,
            [
                [0.227206677, 0.227205575, 0.227228582],
                [0.461318076, 0.461316347, 0.461352766],
                [0.795164824, 0.382332891, 0.28105557],
                [0.598036408, 0.724338114, 0.953801811],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_base_contrast",
            0.75,
            2.0,
            [
                [0.555030465, 0.555029154, 0.555055737],
                [0.756242633, 0.756241679, 0.756263971],
                [0.933571935, 0.708191395, 0.645189881],
                [0.83350718, 0.891457677, 0.992910445],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_medium_low_contrast",
            0.0,
            1.0,
            [
                [0.245546639, 0.245545477, 0.245570093],
                [0.461316764, 0.461314797, 0.461351395],
                [0.778202653, 0.38646546, 0.290383846],
                [0.571873009, 0.703091681, 0.934023201],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_medium_low_contrast",
            0.75,
            2.0,
            [
                [0.567413688, 0.567412019, 0.567438841],
                [0.749255955, 0.749254942, 0.749277532],
                [0.922169328, 0.702421188, 0.6402179],
                [0.815943122, 0.87781173, 0.982940793],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_low_contrast",
            0.0,
            1.0,
            [
                [0.264489919, 0.264488459, 0.264514655],
                [0.461317182, 0.461315215, 0.461351395],
                [0.759039462, 0.391788512, 0.301928937],
                [0.549826682, 0.681571782, 0.909661233],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_low_contrast",
            0.75,
            2.0,
            [
                [0.579899669, 0.579898536, 0.579925299],
                [0.74192977, 0.741928637, 0.741951883],
                [0.908426762, 0.69777, 0.63735646],
                [0.797978163, 0.863464832, 0.970741153],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_very_low_contrast",
            0.0,
            1.0,
            [
                [0.28565523, 0.28565374, 0.285681546],
                [0.46131748, 0.461315691, 0.461351693],
                [0.736840189, 0.398169011, 0.315924495],
                [0.530825257, 0.660286903, 0.880520761],
            ],
        ),
        probed(
            ViewTransform::AgX,
            "agx_very_low_contrast",
            0.75,
            2.0,
            [
                [0.592128515, 0.592127144, 0.592154384],
                [0.734535336, 0.734534204, 0.734557748],
                [0.893283725, 0.6933285, 0.636234879],
                [0.780492544, 0.847702265, 0.955794811],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_punchy",
            0.0,
            1.0,
            [
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.472503841],
                [0.0, 1.0, 0.472503841],
                [1.0, 1.0, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_punchy",
            0.75,
            2.0,
            [
                [0.0, 1.0, 1.0],
                [0.687387466, 0.687387466, 0.687387466],
                [0.687387466, 0.687387466, 0.687387466],
                [1.0, 1.0, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_greyscale",
            0.0,
            1.0,
            [
                [0.0, 1.0, 1.0],
                [0.472503841, 0.472503841, 0.472503841],
                [0.472503841, 1.0, 0.0],
                [1.0, 0.472503841, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_greyscale",
            0.75,
            2.0,
            [
                [0.0, 1.0, 0.687387466],
                [0.687387466, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.687387466, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_very_high_contrast",
            0.0,
            1.0,
            [
                [0.0, 1.0, 1.0],
                [0.472503841, 0.472503841, 0.472503841],
                [0.472503841, 1.0, 0.0],
                [1.0, 0.472503841, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_very_high_contrast",
            0.75,
            2.0,
            [
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.687387466, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_high_contrast",
            0.0,
            1.0,
            [
                [0.0, 1.0, 1.0],
                [0.472503841, 0.472503841, 0.472503841],
                [0.472503841, 1.0, 0.0],
                [1.0, 0.472503841, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_high_contrast",
            0.75,
            2.0,
            [
                [0.0, 1.0, 0.687387466],
                [0.687387466, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.687387466, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_medium_high_contrast",
            0.0,
            1.0,
            [
                [0.0, 1.0, 1.0],
                [0.472503841, 0.472503841, 0.472503841],
                [0.472503841, 0.472503841, 0.472503841],
                [1.0, 0.472503841, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_medium_high_contrast",
            0.75,
            2.0,
            [
                [0.0, 1.0, 0.687387466],
                [0.687387466, 1.0, 0.0],
                [0.687387466, 1.0, 0.0],
                [1.0, 0.687387466, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_base_contrast",
            0.0,
            1.0,
            [
                [0.0, 1.0, 1.0],
                [0.472503841, 0.472503841, 0.472503841],
                [0.472503841, 0.472503841, 0.472503841],
                [1.0, 1.0, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_base_contrast",
            0.75,
            2.0,
            [
                [0.0, 1.0, 0.687387466],
                [0.687387466, 1.0, 0.0],
                [0.687387466, 1.0, 0.0],
                [1.0, 0.687387466, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_medium_low_contrast",
            0.0,
            1.0,
            [
                [0.0, 1.0, 1.0],
                [0.472503841, 0.472503841, 0.472503841],
                [0.472503841, 0.472503841, 0.472503841],
                [1.0, 1.0, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_medium_low_contrast",
            0.75,
            2.0,
            [
                [0.0, 1.0, 0.687387466],
                [0.687387466, 1.0, 0.0],
                [0.687387466, 1.0, 0.0],
                [1.0, 0.687387466, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_low_contrast",
            0.0,
            1.0,
            [
                [0.0, 1.0, 0.472503841],
                [0.472503841, 0.472503841, 0.472503841],
                [0.472503841, 0.472503841, 0.472503841],
                [1.0, 1.0, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_low_contrast",
            0.75,
            2.0,
            [
                [0.0, 1.0, 0.687387466],
                [0.687387466, 1.0, 0.0],
                [0.687387466, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_very_low_contrast",
            0.0,
            1.0,
            [
                [0.0, 1.0, 0.472503841],
                [0.472503841, 0.472503841, 0.472503841],
                [0.472503841, 0.472503841, 0.472503841],
                [1.0, 1.0, 0.0],
            ],
        ),
        probed(
            ViewTransform::FalseColor,
            "false_color_very_low_contrast",
            0.75,
            2.0,
            [
                [0.0, 1.0, 0.687387466],
                [0.687387466, 1.0, 0.0],
                [0.687387466, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
        ),
    ];

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

    #[test]
    fn luminance_compensation_lut_parses_as_37_cubed() -> Result<(), Box<dyn std::error::Error>> {
        let lut = parse_cube(LUMINANCE_COMPENSATION_CUBE)?;
        assert_eq!(lut.size, 37);
        assert_eq!(lut.data.len(), 37 * 37 * 37);
        Ok(())
    }

    #[test]
    fn filmic_look_curves_parse_with_4096_entries() -> Result<(), Box<dyn std::error::Error>> {
        for source in [
            FILMIC_TO_1_20_SPI1D,
            FILMIC_TO_0_99_SPI1D,
            FILMIC_TO_0_85_SPI1D,
            FILMIC_TO_0_60_SPI1D,
            FILMIC_TO_0_48_SPI1D,
            FILMIC_TO_0_35_SPI1D,
        ] {
            assert_eq!(parse_spi1d(source)?.rows.len(), 4096);
        }
        Ok(())
    }

    #[test]
    fn composed_filmic_look_table_has_65536_monotone_entries() -> Result<(), String> {
        let table = filmic_look_table(FilmicCurve::VeryHigh).map_err(|error| error.to_string())?;
        assert_eq!(table.len(), COMPOSED_LUT_SIZE);
        assert!(table.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(table[0] >= 0.0 && table[COMPOSED_LUT_SIZE - 1] <= 1.0);
        Ok(())
    }

    #[test]
    fn find_lut_inv_inverts_a_monotone_table() {
        let table: Vec<f32> = (0..=4).map(|index| index as f32 / 4.0).collect();
        for step in 0..=40 {
            let value = step as f32 / 40.0;
            assert!(
                (find_lut_inv(&table, value) - value).abs() < 1.0e-6,
                "{value}"
            );
        }
    }

    #[test]
    fn looks_match_blender_probe() -> Result<(), Box<dyn std::error::Error>> {
        for row in PROBED {
            let look = Look::parse(row.look).ok_or_else(|| format!("unknown look {}", row.look))?;
            let tolerance = probed_tolerance(row.view);
            for (sample, expected) in SAMPLES.iter().zip(row.expected) {
                let encoded = display_view(row.view, look, row.exposure, *sample)?;
                let graded = if row.gamma != 1.0 {
                    let exponent = (1.0 / row.gamma) as f32;
                    encoded.map(|value| fast_pow(value, exponent))
                } else {
                    encoded
                };
                for channel in 0..3 {
                    let value = f64::from(graded[channel].clamp(0.0, 1.0));
                    assert!(
                        (value - expected[channel]).abs() <= tolerance,
                        "{} {:?} gamma {} sample {sample:?}: {value} vs {}",
                        row.look,
                        row.view,
                        row.gamma,
                        expected[channel]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn incompatible_look_is_rejected() {
        let rejected = display_view(ViewTransform::AgX, Look::VeryHighContrast, 0.0, [0.18; 3]);
        assert!(rejected.is_err());
    }
}
