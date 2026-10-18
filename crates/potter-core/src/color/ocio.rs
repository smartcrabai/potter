//! Blender 5.2.2 OpenColorIO view transforms.
//!
//! Ports the `Standard`, `AgX Base Rec.1886`, `Filmic sRGB`, and `AgX False Color Rec.709` views of
//! Blender 5.2.2's bundled OCIO config (OCIO 2.5.0, display `sRGB`, look `None`). Sampling follows
//! OpenColorIO 2.5.0 branch for branch: `lg2` and `uniform` allocations without clamping, tetrahedral
//! 3D LUT interpolation (`Lut3DOpCPU`), and linear 1D LUT interpolation (`Lut1DOpCPU`), with the
//! coordinate clamped to the LUT domain.
//!
//! Vendored LUTs in `color/luts/` are byte-for-byte copies of Blender 5.2.2's
//! `release/datafiles/colormanagement/luts/` and `release/datafiles/colormanagement/filmic/`
//! (blender/blender tag v5.2.2):
//! - `AgX_Base_sRGB.cube`, `AgX_Base_Rec2020.cube`, `AgX_False_Color.spi1d`: AgX by Troy Sobotka and EaryChow.
//! - `filmic_desat_33.cube`, `filmic_to_0-70_1-03.spi1d`: Filmic by Troy Sobotka.
//!
//! Attribution, file checksums, and licence status are in `color/luts/NOTICE`. The LUT terms the
//! upstream AgX authors attached are unverified there; Blender's own distribution is GPL-3.0-or-later.

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

/// `lg2` allocation window of the AgX views (`AgX Base Rec.1886`, `AgX False Color Rec.709`).
const AGX_LOG2_MIN: f64 = -12.473_93;
const AGX_LOG2_MAX: f64 = 12.526_068_811_7;
/// `lg2` allocation window of the Filmic view (`Filmic sRGB`), span 25.0.
const FILMIC_LOG2_MIN: f64 = -12.473_931_188;
const FILMIC_LOG2_MAX: f64 = 12.526_068_812;
/// `uniform` allocation upper bound of the Filmic desaturation LUT output.
const FILMIC_UNIFORM_MAX: f64 = 0.66;

const REC1886_EXPONENT: f64 = 2.4;
const FALSE_COLOR_EXPONENT: f64 = 2.5;

/// Rec.709 linear to FilmLight E-Gamut linear: `XYZ(D65)->E-Gamut * inverse(XYZ(D65)->Rec.709)`,
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

/// AgX display-linear for `AgX Base Rec.1886`: E-Gamut, `lg2` window, tetrahedral `AgX_Base_sRGB`
/// lookup, then the Rec.1886 decode (`^2.4`). The sRGB encode of the result returns the LUT output.
pub(super) fn agx_display_linear(rgb: [f64; 3]) -> Result<[f64; 3]> {
    let lut = loaded(LazyLock::force(&AGX_BASE_SRGB))?;
    Ok(lut.sample(agx_position(rgb)).map(rec1886_decode))
}

/// Filmic display-linear for `Filmic sRGB`: `lg2` window, tetrahedral `filmic_desat_33` lookup,
/// `uniform` allocation over `[0, 0.66]`, the 1D `filmic_to_0-70_1-03` curve, then an sRGB decode so
/// that the sRGB encode returns the curve output. Input is already Rec.709, so no ACES leg applies.
pub(super) fn filmic_display_linear(rgb: [f64; 3]) -> Result<[f64; 3]> {
    let desaturation = loaded(LazyLock::force(&FILMIC_DESAT_33))?;
    let curve = loaded(LazyLock::force(&FILMIC_TO_0_70))?;
    let desaturated =
        desaturation.sample(rgb.map(|value| lg2_index(value, FILMIC_LOG2_MIN, FILMIC_LOG2_MAX)));
    Ok(desaturated.map(|value| {
        let encoded = curve.sample(value / FILMIC_UNIFORM_MAX)[0];
        srgb_to_linear(encoded)
    }))
}

/// AgX display-linear for `AgX False Color Rec.709`: E-Gamut and `lg2` window, tetrahedral Rec.2020
/// `AgX_Base` lookup, Rec.2020 linear luma, `^(1/2.5)`, then the per-channel `AgX_False_Color`
/// palette and the Rec.1886 decode.
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

/// `lg2` allocation forward (`log2`, then linear map of `[min, max]` to `[0, 1]`) with the clamp
/// applied by the view transforms. NaN and non-positive inputs map to 0.
fn lg2_index(value: f64, min: f64, max: f64) -> f64 {
    if value.is_nan() || value <= 0.0 {
        return 0.0;
    }
    ((value.log2() - min) / (max - min)).clamp(0.0, 1.0)
}

/// Clamps `value` to the lattice `[0, last]` (NaN maps to 0) and splits it into the lower and upper
/// node indices and the fractional weight, as OpenColorIO's LUT ops do.
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

    /// Tetrahedral interpolation, branch for branch with OpenColorIO 2.5.0 `Lut3DOpCPU`.
    fn sample(&self, rgb: [f64; 3]) -> [f64; 3] {
        let last = self.size - 1;
        let (r0, r1, fx) = lattice_position(rgb[0], last);
        let (g0, g1, fy) = lattice_position(rgb[1], last);
        let (b0, b1, fz) = lattice_position(rgb[2], last);
        let n000 = self.node(r0, g0, b0);
        let n100 = self.node(r1, g0, b0);
        let n010 = self.node(r0, g1, b0);
        let n001 = self.node(r0, g0, b1);
        let n110 = self.node(r1, g1, b0);
        let n101 = self.node(r1, g0, b1);
        let n011 = self.node(r0, g1, b1);
        let n111 = self.node(r1, g1, b1);
        if fx > fy {
            if fy > fz {
                weighted([
                    (1.0 - fx, n000),
                    (fx - fy, n100),
                    (fy - fz, n110),
                    (fz, n111),
                ])
            } else if fx > fz {
                weighted([
                    (1.0 - fx, n000),
                    (fx - fz, n100),
                    (fz - fy, n101),
                    (fy, n111),
                ])
            } else {
                weighted([
                    (1.0 - fz, n000),
                    (fz - fx, n001),
                    (fx - fy, n101),
                    (fy, n111),
                ])
            }
        } else if fz > fy {
            weighted([
                (1.0 - fz, n000),
                (fz - fy, n001),
                (fy - fx, n011),
                (fx, n111),
            ])
        } else if fz > fx {
            weighted([
                (1.0 - fy, n000),
                (fy - fz, n010),
                (fz - fx, n011),
                (fx, n111),
            ])
        } else {
            weighted([
                (1.0 - fy, n000),
                (fy - fx, n010),
                (fx - fz, n110),
                (fz, n111),
            ])
        }
    }
}

/// 1D LUT over the domain `[0, 1]`; single-component files are broadcast to three channels.
struct Lut1d {
    rows: Vec<[f64; 3]>,
}

impl Lut1d {
    /// Linear interpolation, matching OpenColorIO 2.5.0 `Lut1DOpCPU`.
    fn sample(&self, value: f64) -> [f64; 3] {
        let (low, high, fraction) = lattice_position(value, self.rows.len() - 1);
        std::array::from_fn(|channel| {
            self.rows[low][channel] * (1.0 - fraction) + self.rows[high][channel] * fraction
        })
    }
}

/// Maps a lazily parsed LUT load failure to an internal error; the vendored data is compiled in,
/// so a failure is a build defect rather than user input.
fn loaded<T>(lut: &'static Result<T, String>) -> Result<&'static T> {
    lut.as_ref().map_err(|message| {
        PotError::new(
            ErrorCode::InternalError,
            format!("vendored OCIO LUT failed to load: {message}"),
        )
    })
}

const TOLERANCE: f64 = 1.0e-12;

fn parse_number(token: &str) -> Result<f64, String> {
    token
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("invalid number `{token}`"))
}

fn parse_triple(tokens: &[&str]) -> Result<[f64; 3], String> {
    match tokens {
        [red, green, blue] => Ok([
            parse_number(red)?,
            parse_number(green)?,
            parse_number(blue)?,
        ]),
        _ => Err(format!(
            "expected three values, found `{}`",
            tokens.join(" ")
        )),
    }
}

/// Parses a `.cube` 3D LUT with `LUT_3D_SIZE N` and exactly `N^3` data rows. `TITLE` and `#`
/// comment lines are ignored. `DOMAIN_MIN 0 0 0` and `DOMAIN_MAX 1 1 1` must match when present;
/// when absent OpenColorIO uses that domain by default, and `filmic_desat_33.cube` omits them.
fn parse_cube(source: &str) -> Result<Lut3d, String> {
    let mut size = None;
    let mut data = Vec::new();
    for line in source.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') || line.starts_with("TITLE ") {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.as_slice() {
            ["LUT_3D_SIZE", value] => {
                if size.is_some() {
                    return Err("duplicate LUT_3D_SIZE".to_owned());
                }
                let parsed = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid LUT_3D_SIZE `{value}`: {error}"))?;
                if parsed < 2 {
                    return Err(format!("LUT_3D_SIZE must be at least 2, found {parsed}"));
                }
                size = Some(parsed);
            }
            ["DOMAIN_MIN", rest @ ..] => expect_domain(rest, 0.0)?,
            ["DOMAIN_MAX", rest @ ..] => expect_domain(rest, 1.0)?,
            [_, _, _] => {
                if size.is_none() {
                    return Err("data row before LUT_3D_SIZE".to_owned());
                }
                data.push(parse_triple(&tokens)?);
            }
            _ => return Err(format!("unrecognised cube line `{line}`")),
        }
    }
    let size = size.ok_or_else(|| "missing LUT_3D_SIZE".to_owned())?;
    let expected = size
        .checked_pow(3)
        .ok_or_else(|| format!("LUT_3D_SIZE {size} is too large"))?;
    if data.len() != expected {
        return Err(format!(
            "expected {expected} LUT entries, found {}",
            data.len()
        ));
    }
    Ok(Lut3d { size, data })
}

fn expect_domain(tokens: &[&str], expected: f64) -> Result<(), String> {
    let domain = parse_triple(tokens)?;
    if domain
        .iter()
        .any(|value| (value - expected).abs() > TOLERANCE)
    {
        return Err(format!(
            "unsupported LUT domain `{}`; expected {expected} per channel",
            tokens.join(" ")
        ));
    }
    Ok(())
}

/// Parses a `.spi1d` 1D LUT: requires `From 0.0 1.0`, `Length` (at least 2), and `Components 1|3`,
/// with exactly `Length` data rows of `Components` values.
fn parse_spi1d(source: &str) -> Result<Lut1d, String> {
    let mut domain_seen = false;
    let mut length = None;
    let mut components = None;
    let mut rows = Vec::new();
    for line in source.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.as_slice() {
            ["Version", "1"] | ["{"] | ["}"] => {}
            ["From", low, high] => {
                let low = parse_number(low)?;
                let high = parse_number(high)?;
                if low.abs() > TOLERANCE || (high - 1.0).abs() > TOLERANCE {
                    return Err(format!("unsupported spi1d domain `{line}`"));
                }
                domain_seen = true;
            }
            ["Length", value] => {
                let parsed = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid Length `{value}`: {error}"))?;
                if parsed < 2 {
                    return Err(format!("Length must be at least 2, found {parsed}"));
                }
                length = Some(parsed);
            }
            ["Components", value] => {
                let parsed = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid Components `{value}`: {error}"))?;
                if parsed != 1 && parsed != 3 {
                    return Err(format!("Components must be 1 or 3, found {parsed}"));
                }
                components = Some(parsed);
            }
            _ => {
                let row = match tokens.as_slice() {
                    [value] if components == Some(1) => [parse_number(value)?; 3],
                    [_, _, _] if components == Some(3) => parse_triple(&tokens)?,
                    _ => return Err(format!("row `{line}` does not match Components")),
                };
                rows.push(row);
            }
        }
    }
    if !domain_seen {
        return Err("missing `From 0.0 1.0` header".to_owned());
    }
    let length = length.ok_or_else(|| "missing Length header".to_owned())?;
    if components.is_none() {
        return Err("missing Components header".to_owned());
    }
    if rows.len() != length {
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
