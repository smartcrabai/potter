//! Blender 5.2.2 `OpenColorIO` display views for the `sRGB` display.
//!
//! Ports the `AgX Base Rec.1886`, `Filmic sRGB`, and `AgX False Color Rec.709` views of Blender
//! 5.2.2's bundled OCIO config (`OpenColorIO` 2.5.0). Sampling follows `OpenColorIO` 2.5.0: the
//! `lg2` allocation has no clamp (callers clamp to the LUT domain), 3D LUTs use tetrahedral
//! interpolation (`Lut3DOpCPU`), and 1D LUTs use linear interpolation (`Lut1DOpCPU`).
//!
//! The LUTs in `color/luts/` are byte-for-byte copies of Blender 5.2.2 data. Source, SHA-256,
//! authors, and licence status are recorded in `color/luts/NOTICE`.

use std::sync::LazyLock;

use super::srgb_to_linear;
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

/// `lg2` allocation window of the `AgX` views (`AgX Base Rec.1886`, `AgX False Color Rec.709`).
const AGX_LOG2_MIN: f64 = -12.473_93;
const AGX_LOG2_MAX: f64 = 12.526_068_811_7;
/// `lg2` allocation window of the Filmic view (`Filmic sRGB`), span 25.0.
const FILMIC_LOG2_MIN: f64 = -12.473_931_188;
const FILMIC_LOG2_MAX: f64 = 12.526_068_812;
/// `uniform` allocation upper bound of the Filmic desaturation LUT output.
const FILMIC_UNIFORM_MAX: f64 = 0.66;

const REC1886_EXPONENT: f64 = 2.4;
const FALSE_COLOR_EXPONENT: f64 = 2.5;

/// Rec.709 linear to `FilmLight` E-Gamut linear: `XYZ(D65)->E-Gamut * inverse(XYZ(D65)->Rec.709)`,
/// computed numerically from the Blender 5.2.2 config matrices `XYZ (D65) -> Linear Rec.709` and
/// `XYZ (D65) -> Linear FilmLight E-Gamut`, rounded to 12 decimals. Row sums are 1 within 1e-7.
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

/// `AgX Base Rec.1886` display-linear: E-Gamut, `lg2` window, tetrahedral `AgX_Base_sRGB` lookup
/// (its output is Rec.1886 encoded), then the 2.4 power decode. The caller's sRGB encode is the
/// display value.
pub(super) fn agx_display_linear(rgb: [f64; 3]) -> Result<[f64; 3]> {
    let lut = loaded(LazyLock::force(&AGX_BASE_SRGB))?;
    Ok(lut.sample(agx_position(rgb)).map(rec1886_decode))
}

/// `Filmic sRGB` display-linear: `lg2` window, tetrahedral `filmic_desat_33` lookup, `uniform`
/// allocation over `[0, 0.66]`, the 1D `filmic_to_0-70_1-03` curve, then an sRGB decode so that
/// the caller's sRGB encode returns the curve output. The input is already Rec.709, so no ACES leg
/// applies.
pub(super) fn filmic_display_linear(rgb: [f64; 3]) -> Result<[f64; 3]> {
    let desaturation = loaded(LazyLock::force(&FILMIC_DESAT_33))?;
    let curve = loaded(LazyLock::force(&FILMIC_TO_0_70))?;
    let desaturated =
        desaturation.sample(rgb.map(|value| lg2_index(value, FILMIC_LOG2_MIN, FILMIC_LOG2_MAX)));
    Ok(desaturated.map(|value| srgb_to_linear(curve.sample(value / FILMIC_UNIFORM_MAX)[0])))
}

/// `AgX False Color Rec.709` display-linear: E-Gamut and `lg2` window, tetrahedral Rec.2020
/// `AgX_Base` lookup, Rec.2020 linear luma, `^(1/2.5)`, the per-channel `AgX_False_Color` palette,
/// then the 2.4 power decode.
pub(super) fn false_color_display_linear(rgb: [f64; 3]) -> Result<[f64; 3]> {
    let lut = loaded(LazyLock::force(&AGX_BASE_REC2020))?;
    let palette = loaded(LazyLock::force(&AGX_FALSE_COLOR))?;
    let linear = lut.sample(agx_position(rgb)).map(rec1886_decode);
    let luma: f64 = FALSE_COLOR_LUMA
        .iter()
        .zip(linear)
        .map(|(weight, value)| weight * value)
        .sum();
    let position = if luma > 0.0 {
        luma.powf(1.0 / FALSE_COLOR_EXPONENT)
    } else {
        0.0
    };
    Ok(palette.sample(position).map(rec1886_decode))
}

fn agx_position(rgb: [f64; 3]) -> [f64; 3] {
    rec709_to_e_gamut(rgb).map(|value| lg2_index(value, AGX_LOG2_MIN, AGX_LOG2_MAX))
}

fn rec709_to_e_gamut(rgb: [f64; 3]) -> [f64; 3] {
    REC709_TO_E_GAMUT.map(|row| row[0] * rgb[0] + row[1] * rgb[1] + row[2] * rgb[2])
}

fn rec1886_decode(value: f64) -> f64 {
    value.max(0.0).powf(REC1886_EXPONENT)
}

/// `lg2` allocation forward: `log2`, then a linear map of `[min, max]` to `[0, 1]`, clamped to
/// `[0, 1]` as the LUT domain requires. NaN and non-positive inputs map to 0.
fn lg2_index(value: f64, min: f64, max: f64) -> f64 {
    if value.is_nan() || value <= 0.0 {
        return 0.0;
    }
    ((value.log2() - min) / (max - min)).clamp(0.0, 1.0)
}

/// Clamps `value` to the lattice `[0, last]` (NaN maps to 0) and splits it into the lower and upper
/// node indices and the fractional weight.
fn lattice_position(value: f64, last: usize) -> (usize, usize, f64) {
    let last = last as f64;
    let position = if value.is_nan() {
        0.0
    } else {
        (value * last).clamp(0.0, last)
    };
    let low = position.floor();
    (low as usize, position.ceil() as usize, position - low)
}

fn weighted(terms: [(f64, [f64; 3]); 4]) -> [f64; 3] {
    std::array::from_fn(|channel| {
        terms
            .iter()
            .map(|(weight, value)| weight * value[channel])
            .sum::<f64>()
    })
}

/// Cube 3D LUT with `size^3` nodes stored in file order (red fastest).
struct Lut3d {
    size: usize,
    data: Vec<[f64; 3]>,
}

impl Lut3d {
    fn node(&self, red: usize, green: usize, blue: usize) -> [f64; 3] {
        self.data[red + self.size * (green + self.size * blue)]
    }

    /// Tetrahedral interpolation, branch for branch with `OpenColorIO` 2.5.0 `Lut3DOpCPU`. Each
    /// branch reads only the four corners it blends.
    fn sample(&self, rgb: [f64; 3]) -> [f64; 3] {
        let last = self.size - 1;
        let (r0, r1, fx) = lattice_position(rgb[0], last);
        let (g0, g1, fy) = lattice_position(rgb[1], last);
        let (b0, b1, fz) = lattice_position(rgb[2], last);
        let corner = |red: bool, green: bool, blue: bool| {
            self.node(
                if red { r1 } else { r0 },
                if green { g1 } else { g0 },
                if blue { b1 } else { b0 },
            )
        };
        if fx > fy {
            if fy > fz {
                weighted([
                    (1.0 - fx, corner(false, false, false)),
                    (fx - fy, corner(true, false, false)),
                    (fy - fz, corner(true, true, false)),
                    (fz, corner(true, true, true)),
                ])
            } else if fx > fz {
                weighted([
                    (1.0 - fx, corner(false, false, false)),
                    (fx - fz, corner(true, false, false)),
                    (fz - fy, corner(true, false, true)),
                    (fy, corner(true, true, true)),
                ])
            } else {
                weighted([
                    (1.0 - fz, corner(false, false, false)),
                    (fz - fx, corner(false, false, true)),
                    (fx - fy, corner(true, false, true)),
                    (fy, corner(true, true, true)),
                ])
            }
        } else if fz > fy {
            weighted([
                (1.0 - fz, corner(false, false, false)),
                (fz - fy, corner(false, false, true)),
                (fy - fx, corner(false, true, true)),
                (fx, corner(true, true, true)),
            ])
        } else if fz > fx {
            weighted([
                (1.0 - fy, corner(false, false, false)),
                (fy - fz, corner(false, true, false)),
                (fz - fx, corner(false, true, true)),
                (fx, corner(true, true, true)),
            ])
        } else {
            weighted([
                (1.0 - fy, corner(false, false, false)),
                (fy - fx, corner(false, true, false)),
                (fx - fz, corner(true, true, false)),
                (fz, corner(true, true, true)),
            ])
        }
    }
}

/// 1D LUT over the domain `[0, 1]`; single-component files are broadcast to three channels.
struct Lut1d {
    rows: Vec<[f64; 3]>,
}

impl Lut1d {
    /// Linear interpolation, matching `OpenColorIO` 2.5.0 `Lut1DOpCPU`.
    fn sample(&self, value: f64) -> [f64; 3] {
        let (low, high, fraction) = lattice_position(value, self.rows.len() - 1);
        std::array::from_fn(|channel| {
            self.rows[low][channel] * (1.0 - fraction) + self.rows[high][channel] * fraction
        })
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

fn parse_triple(values: &[f64]) -> [f64; 3] {
    [values[0], values[1], values[2]]
}

/// Parses a `.cube` 3D LUT: the `LUT_3D_SIZE N` header and exactly `N^3` rows of three numbers.
/// Other lines (`TITLE`, `DOMAIN_*`, comments) are skipped; the vendored files use the `[0, 1]`
/// domain.
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
        match parse_numbers(&tokens).as_deref() {
            Ok([value]) => rows.push([*value; 3]),
            Ok([red, green, blue]) => rows.push([*red, *green, *blue]),
            _ => {}
        }
    }
    let length = length.ok_or_else(|| "missing Length header".to_owned())?;
    if length < 2 || rows.len() != length {
        return Err(format!("expected {length} rows, found {}", rows.len()));
    }
    Ok(Lut1d { rows })
}

fn parse_numbers(tokens: &[&str]) -> std::result::Result<Vec<f64>, std::num::ParseFloatError> {
    tokens.iter().map(|token| token.parse::<f64>()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tetrahedral_sampling_returns_node_values_at_lattice_points() {
        let size = 3;
        let data: Vec<[f64; 3]> = (0..size)
            .flat_map(|blue| {
                (0..size).flat_map(move |green| {
                    (0..size)
                        .map(move |red| [red as f64 / 2.0, green as f64 / 2.0, blue as f64 / 2.0])
                })
            })
            .collect();
        let lut = Lut3d { size, data };
        for blue in 0..size {
            for green in 0..size {
                for red in 0..size {
                    let expected = [red as f64 / 2.0, green as f64 / 2.0, blue as f64 / 2.0];
                    let sampled = lut.sample(expected);
                    for channel in 0..3 {
                        assert!((sampled[channel] - expected[channel]).abs() < 1.0e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn rec709_to_e_gamut_maps_white_to_white() {
        for channel in rec709_to_e_gamut([1.0; 3]) {
            assert!((channel - 1.0).abs() < 1.0e-6);
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
