use serde::{Deserialize, Serialize};
use std::{borrow::Cow, collections::BTreeMap};

use crate::color::linear_to_srgb_unclamped;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageInterpolation {
    #[default]
    Linear,
    Closest,
}

/// How an image lookup past its edge is resolved, as Blender's Image Texture `extension`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageExtension {
    #[default]
    Repeat,
    Extend,
    Clip,
    Mirror,
}

impl ImageExtension {
    /// Whether this is the default REPEAT, which scene JSON omits.
    #[must_use]
    pub const fn is_repeat(&self) -> bool {
        matches!(self, Self::Repeat)
    }
}

/// One UDIM tile. `pixels` are linear light, and `srgb` is as in [`ImageData`].
#[derive(Clone, Debug, PartialEq)]
pub struct ImageTileData {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f64; 4]>,
    pub srgb: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    /// The values as loaded: straight colour for a straight image, premultiplied for a premultiplied one, and raw for
    /// data. The image sampler applies the alpha rule from [`ImageStorage`]; direct readers call
    /// [`ImageData::straight_pixels`].
    pub pixels: Vec<[f64; 4]>,
    pub tiles: BTreeMap<u32, ImageTileData>,
    pub interpolation: ImageInterpolation,
    /// Whether `pixels` were decoded from sRGB-encoded storage. Linear sampling then interpolates the encoded values
    /// and decodes the result, as Cycles does for images it compresses as sRGB (`svm_image_texture` with
    /// `NODE_IMAGE_COMPRESS_AS_SRGB`, set from `ImageMetaData::is_compressible_as_srgb`).
    pub srgb: bool,
    /// How Blender stores the image's values for Image Texture colour reads.
    pub storage: ImageStorage,
}

impl ImageData {
    /// The colour as straight values, for readers that do not go through the sampler (export, compositing, simulation
    /// textures, environment importance). A straight or data image is returned as stored. A premultiplied colour image
    /// has its RGB divided by alpha, and zero where alpha is zero; that is the only conversion. Tiles are not converted.
    #[must_use]
    pub fn straight_pixels(&self) -> Cow<'_, [[f64; 4]]> {
        if self.storage.ignore_alpha {
            // Alpha ignored: the colour is as stored and every alpha is 1 (`ImageStorage::ignore_alpha`).
            return Cow::Owned(
                self.pixels
                    .iter()
                    .map(|pixel| [pixel[0], pixel[1], pixel[2], 1.0])
                    .collect(),
            );
        }
        if self.storage.straight_alpha || self.storage.data {
            return Cow::Borrowed(&self.pixels);
        }
        Cow::Owned(
            self.pixels
                .iter()
                .map(|pixel| {
                    let alpha = pixel[3];
                    if alpha == 0.0 {
                        [0.0, 0.0, 0.0, alpha]
                    } else {
                        [pixel[0] / alpha, pixel[1] / alpha, pixel[2] / alpha, alpha]
                    }
                })
                .collect(),
        )
    }
}

/// The precision a stored image was quantised to. Blender premultiplies straight alpha at load in the arithmetic of the
/// buffer it loads (Cycles' `util_image_multiply_native`, `intern/cycles/util/image.h`): integer `c * a / max` for 8-bit
/// and 16-bit buffers, which truncates; a half product rounded to half for half buffers; and a plain product for float.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    Byte,
    Short,
    Half,
    #[default]
    Float,
}

/// How Blender stores an image for Image Texture colour reads, from its colour space, alpha mode, and bit depth.
/// Cycles' image manager premultiplies straight alpha for a non-data image when it loads it, so a colour read whose
/// Alpha output is unlinked returns premultiplied values. Verified against the Blender 5.2.2 renders in
/// `tests/fixtures/alpha_texture` for sRGB and Linear 8-bit, 16-bit, half and float EXR, TIFF, and Non-Color images.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImageStorage {
    /// A non-colour (data) image: values are read as stored, with no premultiply and no unassociate.
    pub data: bool,
    /// Straight alpha, which Blender premultiplies at load for a non-data image.
    pub straight_alpha: bool,
    /// Alpha ignored (Blender's `IMA_ALPHA_IGNORE`, a `None` alpha mode): the loader sets alpha to 1 for an RGBA image
    /// (`image_metadata.cpp`, `ignore_alpha`), no premultiply happens, and no colour is unassociated (`shader_nodes.cpp`
    /// leaves IGNORE out of the unassociate flag). Every read returns alpha 1.
    pub ignore_alpha: bool,
    /// The precision the premultiply rounds to.
    pub precision: Precision,
}

const TRANSPARENT_BLACK: [f64; 4] = [0.0; 4];

/// How an sRGB image's stored values are mixed and then read (Cycles' `svm_image_texture` and Blender's image
/// buffer reader). Linear images and the other transfers use [`Transfer::Linear`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transfer {
    /// Values are mixed as stored, with no encoding step.
    Linear,
    /// sRGB values are mixed encoded, then decoded to linear light.
    SrgbDecoded,
    /// sRGB values are mixed encoded and returned encoded, as Blender's intensity channel reads them.
    SrgbEncoded,
}

/// Which read of the stored image a sample makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Read {
    /// A modifier mask (`BKE_texture_get_value`): straight values, decoded to linear light for colour channels.
    Decoded,
    /// The mask's intensity channel: straight encoded values, not decoded.
    Encoded,
    /// An Image Texture colour output (`svm_image_texture`). `alpha_linked` is whether the node's Alpha output is linked,
    /// which is when Cycles unassociates the colour.
    Colour { alpha_linked: bool },
}

/// How one sample premultiplies, mixes, and finishes its texels for a [`Read`] of an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Texels {
    transfer: Transfer,
    /// The stored colour is premultiplied by alpha, rounded to this precision, before mixing.
    premultiply: Option<Precision>,
    /// The mixed colour is divided by the mixed alpha, as Cycles unassociates a linked Alpha output.
    unassociate: bool,
    /// Alpha is ignored: every read returns alpha 1 (`ImageStorage::ignore_alpha`).
    ignore_alpha: bool,
}

impl Texels {
    fn new(srgb: bool, storage: ImageStorage, read: Read) -> Self {
        let transfer = match read {
            Read::Encoded if srgb => Transfer::SrgbEncoded,
            Read::Decoded | Read::Colour { .. } if srgb => Transfer::SrgbDecoded,
            _ => Transfer::Linear,
        };
        let colour = matches!(read, Read::Colour { .. });
        // The premultiply happens in the buffer's precision whatever the colour space: 8-bit sRGB and 8-bit linear
        // images stay bytes (`conform_pixels_to_metadata_type`), and only the non-linear, non-sRGB conversions go to half.
        let premultiply =
            (colour && storage.straight_alpha && !storage.data && !storage.ignore_alpha)
                .then_some(storage.precision);
        let unassociate = matches!(read, Read::Colour { alpha_linked: true })
            && !storage.data
            && !storage.ignore_alpha;
        Self {
            transfer,
            premultiply,
            unassociate,
            ignore_alpha: storage.ignore_alpha,
        }
    }
}

/// Samples an image at a normalized UV coordinate, with the colour rule of Blender's texture modifiers: an sRGB
/// image's encoded texels are mixed, and the result is decoded to linear light. Blender's `BKE_texture_get_value`
/// reads the straight encoded bytes (`ibuf_get_color`), mixes them in `boxsample` (a one-texel box, which is the
/// bilinear weighting), and then converts the colour channels with `IMB_colormanagement_colorspace_to_scene_linear_v3`.
/// The modifier reader does not premultiply.
///
/// Tile zero samples the base image directly. Positive tiles are treated as UDIM
/// base tiles, with integer UV offsets selecting neighboring tiles.
#[must_use]
pub fn sample(data: &ImageData, uv: [f64; 2], tile: u32) -> [f64; 4] {
    sample_image(data, uv, tile, data.interpolation, Read::Decoded)
}

/// Samples an image's straight stored values as Blender's intensity channel reads them: an sRGB image's encoded texels
/// are mixed and returned encoded, with no decode. Other images return the same values as [`sample`].
#[must_use]
pub fn sample_encoded(data: &ImageData, uv: [f64; 2], tile: u32) -> [f64; 4] {
    sample_image(data, uv, tile, data.interpolation, Read::Encoded)
}

/// Samples an image's colour as an Image Texture node does, with its Alpha output unlinked: the stored values are
/// premultiplied (for a straight-alpha non-data image) and mixed encoded, then decoded for sRGB. Used by the world
/// environment, whose Alpha output is not wired.
#[must_use]
pub fn sample_with_interpolation(
    data: &ImageData,
    uv: [f64; 2],
    tile: u32,
    interpolation: ImageInterpolation,
) -> [f64; 4] {
    sample_image(
        data,
        uv,
        tile,
        interpolation,
        Read::Colour {
            alpha_linked: false,
        },
    )
}

fn sample_image(
    data: &ImageData,
    uv: [f64; 2],
    tile: u32,
    interpolation: ImageInterpolation,
    read: Read,
) -> [f64; 4] {
    if !uv[0].is_finite() || !uv[1].is_finite() {
        return TRANSPARENT_BLACK;
    }

    let Some(image) = image_for_uv(data, uv, tile) else {
        return TRANSPARENT_BLACK;
    };
    let Some((width, height)) = image.valid_dimensions() else {
        return TRANSPARENT_BLACK;
    };
    let local_uv = if tile == 0 {
        uv
    } else {
        [uv[0] - uv[0].floor(), uv[1] - uv[1].floor()]
    };
    let u = local_uv[0].clamp(0.0, 1.0);
    let v = local_uv[1].clamp(0.0, 1.0);

    let texels = Texels::new(image.srgb, image.storage, read);
    match interpolation {
        ImageInterpolation::Linear => sample_linear(image.pixels, width, height, u, v, texels),
        ImageInterpolation::Closest => sample_closest(image.pixels, width, height, u, v, texels),
    }
}

/// Samples an image as an Image Texture node reads it, with the node's Alpha output unlinked. See
/// [`sample_with_extension_alpha`].
#[must_use]
pub fn sample_with_extension(
    data: &ImageData,
    routing_uv: [f64; 2],
    pixel_uv: [f64; 2],
    tile: u32,
    interpolation: ImageInterpolation,
    extension: ImageExtension,
) -> [f64; 4] {
    sample_with_extension_alpha(
        data,
        routing_uv,
        pixel_uv,
        tile,
        interpolation,
        extension,
        false,
    )
}

/// Samples an image using separate UVs for UDIM routing and pixel lookup, as an Image Texture node reads it.
///
/// `routing_uv` selects the UDIM tile. `pixel_uv` is the full-image UV for tile zero and the
/// tile-local V-down UV for positive base tiles. `alpha_linked` is whether the node's Alpha output is linked, which
/// is when Cycles unassociates the colour (see [`ImageStorage`]).
#[must_use]
pub fn sample_with_extension_alpha(
    data: &ImageData,
    routing_uv: [f64; 2],
    pixel_uv: [f64; 2],
    tile: u32,
    interpolation: ImageInterpolation,
    extension: ImageExtension,
    alpha_linked: bool,
) -> [f64; 4] {
    if !pixel_uv[0].is_finite()
        || !pixel_uv[1].is_finite()
        || (tile != 0 && (!routing_uv[0].is_finite() || !routing_uv[1].is_finite()))
    {
        return TRANSPARENT_BLACK;
    }

    let Some(image) = image_for_uv(data, routing_uv, tile) else {
        return TRANSPARENT_BLACK;
    };
    let Some((width, height)) = image.valid_dimensions() else {
        return TRANSPARENT_BLACK;
    };
    if extension == ImageExtension::Clip
        && (pixel_uv[0] < 0.0 || pixel_uv[0] > 1.0 || pixel_uv[1] < 0.0 || pixel_uv[1] > 1.0)
    {
        return TRANSPARENT_BLACK;
    }

    let local_uv = match extension {
        ImageExtension::Repeat => [pixel_uv[0].rem_euclid(1.0), pixel_uv[1].rem_euclid(1.0)],
        ImageExtension::Extend => [pixel_uv[0].clamp(0.0, 1.0), pixel_uv[1].clamp(0.0, 1.0)],
        ImageExtension::Clip => pixel_uv,
        ImageExtension::Mirror => [pixel_uv[0].rem_euclid(2.0), pixel_uv[1].rem_euclid(2.0)],
    };

    let texels = Texels::new(image.srgb, image.storage, Read::Colour { alpha_linked });
    match interpolation {
        ImageInterpolation::Linear => sample_linear_with_extension(
            image.pixels,
            width,
            height,
            local_uv[0],
            local_uv[1],
            extension,
            texels,
        ),
        ImageInterpolation::Closest => sample_closest_with_extension(
            image.pixels,
            width,
            height,
            local_uv[0],
            local_uv[1],
            extension,
            texels,
        ),
    }
}

fn image_for_uv(data: &ImageData, uv: [f64; 2], tile: u32) -> Option<ImageTileDataRef<'_>> {
    if tile == 0 {
        Some(ImageTileDataRef {
            width: data.width,
            height: data.height,
            pixels: &data.pixels,
            srgb: data.srgb,
            storage: data.storage,
        })
    } else {
        let actual_tile = udim_tile(tile, uv)?;
        let image = data.tiles.get(&actual_tile).map_or_else(
            || {
                if actual_tile == 1001 {
                    ImageTileDataRef {
                        width: data.width,
                        height: data.height,
                        pixels: &data.pixels,
                        srgb: data.srgb,
                        storage: data.storage,
                    }
                } else {
                    ImageTileDataRef {
                        width: 0,
                        height: 0,
                        pixels: &[],
                        srgb: false,
                        storage: ImageStorage::default(),
                    }
                }
            },
            |image| ImageTileDataRef {
                width: image.width,
                height: image.height,
                pixels: &image.pixels,
                srgb: image.srgb,
                storage: data.storage,
            },
        );
        Some(image)
    }
}

struct ImageTileDataRef<'a> {
    width: u32,
    height: u32,
    pixels: &'a [[f64; 4]],
    srgb: bool,
    storage: ImageStorage,
}

impl ImageTileDataRef<'_> {
    fn valid_dimensions(&self) -> Option<(usize, usize)> {
        if self.width == 0 || self.height == 0 {
            return None;
        }
        let width = usize::try_from(self.width).ok()?;
        let height = usize::try_from(self.height).ok()?;
        let pixel_count = width.checked_mul(height)?;
        (self.pixels.len() == pixel_count).then_some((width, height))
    }
}

const I64_UPPER_EXCLUSIVE: f64 = 9_223_372_036_854_775_808.0;
const I64_LOWER_INCLUSIVE: f64 = -9_223_372_036_854_775_808.0;
fn udim_tile(base_tile: u32, uv: [f64; 2]) -> Option<u32> {
    let u_offset = floor_to_i64(uv[0])?;
    let v_offset = floor_to_i64(uv[1])?;
    let offset = v_offset.checked_mul(10)?.checked_add(u_offset)?;
    let actual_tile = i64::from(base_tile).checked_add(offset)?;
    u32::try_from(actual_tile).ok()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the rounded input is checked against exclusive i64 conversion bounds"
)]
fn floor_to_i64(value: f64) -> Option<i64> {
    let value = value.floor();
    // i64::MAX as f64 rounds up to 2^63, so the upper bound is exclusive.

    if !(I64_LOWER_INCLUSIVE..I64_UPPER_EXCLUSIVE).contains(&value) {
        return None;
    }
    Some(value as i64)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "validated image dimensions and clamped UVs keep conversions within exact texel bounds"
)]
fn sample_closest(
    pixels: &[[f64; 4]],
    width: usize,
    height: usize,
    u: f64,
    v: f64,
    texels: Texels,
) -> [f64; 4] {
    let x = ((u * width as f64).floor() as usize).min(width - 1);
    let y = ((v * height as f64).floor() as usize).min(height - 1);
    finish(stored_texel(pixel(pixels, width, x, y), texels), texels)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "validated image dimensions and clamped UVs keep conversions within exact texel bounds"
)]
fn sample_linear(
    pixels: &[[f64; 4]],
    width: usize,
    height: usize,
    u: f64,
    v: f64,
    texels: Texels,
) -> [f64; 4] {
    let x = (u * width as f64 - 0.5).clamp(0.0, (width - 1) as f64);
    let y = (v * height as f64 - 0.5).clamp(0.0, (height - 1) as f64);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;

    let texel = |x: usize, y: usize| stored_texel(pixel(pixels, width, x, y), texels);
    let top = mix(texel(x0, y0), texel(x1, y0), tx);
    let bottom = mix(texel(x0, y1), texel(x1, y1), tx);
    finish(mix(top, bottom, ty), texels)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "validated u32 image dimensions are exactly representable as f64"
)]
fn sample_closest_with_extension(
    pixels: &[[f64; 4]],
    width: usize,
    height: usize,
    u: f64,
    v: f64,
    extension: ImageExtension,
    texels: Texels,
) -> [f64; 4] {
    let Some(mut x) = floor_to_i64(u * width as f64) else {
        return TRANSPARENT_BLACK;
    };
    let Some(mut y) = floor_to_i64(v * height as f64) else {
        return TRANSPARENT_BLACK;
    };
    if extension == ImageExtension::Clip {
        let (Ok(width), Ok(height)) = (i64::try_from(width), i64::try_from(height)) else {
            return TRANSPARENT_BLACK;
        };
        x = x.min(width - 1);
        y = y.min(height - 1);
    }
    finish(
        stored_texel(
            pixel_with_extension(pixels, width, height, x, y, extension),
            texels,
        ),
        texels,
    )
}

#[expect(
    clippy::cast_precision_loss,
    reason = "validated u32 image dimensions are exactly representable as f64"
)]
fn sample_linear_with_extension(
    pixels: &[[f64; 4]],
    width: usize,
    height: usize,
    u: f64,
    v: f64,
    extension: ImageExtension,
    texels: Texels,
) -> [f64; 4] {
    let x = u * width as f64 - 0.5;
    let y = v * height as f64 - 0.5;
    let Some(x0) = floor_to_i64(x) else {
        return TRANSPARENT_BLACK;
    };
    let Some(y0) = floor_to_i64(y) else {
        return TRANSPARENT_BLACK;
    };
    let x1 = x0.saturating_add(1);
    let y1 = y0.saturating_add(1);
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;

    let texel = |x: i64, y: i64| {
        stored_texel(
            pixel_with_extension(pixels, width, height, x, y, extension),
            texels,
        )
    };
    let top = mix(texel(x0, y0), texel(x1, y0), tx);
    let bottom = mix(texel(x0, y1), texel(x1, y1), tx);
    finish(mix(top, bottom, ty), texels)
}

/// One texel as Blender's stored buffer holds it, before mixing. An sRGB image's texel is encoded, which is the value
/// Cycles interpolates (`svm_image_texture`). Straight alpha is premultiplied when the image loaded, in the arithmetic of
/// the buffer's precision. The encode is the exact inverse of the load decode, so an unpremultiplied texel round-trips.
fn stored_texel(linear: [f64; 4], texels: Texels) -> [f64; 4] {
    let mut rgb = [linear[0], linear[1], linear[2]];
    if texels.transfer != Transfer::Linear {
        rgb = rgb.map(linear_to_srgb_unclamped);
    }
    if let Some(precision) = texels.premultiply {
        let alpha = linear[3];
        rgb = rgb.map(|value| premultiply(value, alpha, precision));
    }
    [rgb[0], rgb[1], rgb[2], linear[3]]
}

/// Blender's premultiply at load for one component, in the arithmetic of the buffer it loads
/// (`util_image_multiply_native`, `intern/cycles/util/image.h`). Byte and short buffers multiply the integers `c` and `a`
/// and truncate `c * a / max`, which is what `((uint32_t)a * (uint32_t)b) / 255` does. Half buffers round the product to
/// half, as `float_to_half` does. Float buffers multiply directly.
fn premultiply(value: f64, alpha: f64, precision: Precision) -> f64 {
    match precision {
        Precision::Byte => integer_premultiply(value, alpha, 255.0),
        Precision::Short => integer_premultiply(value, alpha, 65535.0),
        Precision::Half => round_to_half(value * alpha),
        Precision::Float => value * alpha,
    }
}

/// `floor(c * a / max) / max` for the integers `c = value * max` and `a = alpha * max`. A texel decoded from a buffer of
/// this precision has exact integer `c` and `a`, so the product and quotient here are exact integer arithmetic.
fn integer_premultiply(value: f64, alpha: f64, max: f64) -> f64 {
    let colour = (value * max).round();
    let coverage = (alpha * max).round();
    ((colour * coverage) / max).floor() / max
}

/// Rounds to the nearest IEEE half-precision value, ties to even, as Cycles' `float_to_half` does. Normal halves keep
/// 11 significant bits (a step of 2^(e - 10) at exponent e), and values below 2^-14 share the subnormal step 2^-24.
/// Values at or above 65520 overflow to infinity, as they do in half.
fn round_to_half(value: f64) -> f64 {
    if !value.is_finite() || value.abs() < f64::MIN_POSITIVE {
        return value;
    }
    let magnitude = value.abs();
    if magnitude >= 65_520.0 {
        return value.signum() * f64::INFINITY;
    }
    let exponent = i32::try_from((magnitude.to_bits() >> 52) & 0x7ff).unwrap_or(0) - 1023;
    let step = 2.0_f64.powi(exponent.max(-14) - 10);
    value.signum() * ((magnitude / step).round_ties_even() * step)
}

/// The colour a lookup returns from a mixed stored value. Cycles unassociates the colour when the Alpha output is
/// linked (`svm_image_texture`, `NODE_IMAGE_ALPHA_UNASSOCIATE`, only for alpha strictly between 0 and 1), and then
/// decodes an sRGB image to linear light (`color_srgb_to_linear_v4`). Alpha is unchanged.
fn finish(mixed: [f64; 4], texels: Texels) -> [f64; 4] {
    // Alpha ignored (`IMA_ALPHA_IGNORE`): the loader made alpha 1, so every read returns 1 (`image_metadata.cpp`).
    let alpha = if texels.ignore_alpha { 1.0 } else { mixed[3] };
    let mut rgb = [mixed[0], mixed[1], mixed[2]];
    if texels.unassociate && alpha > 0.0 && alpha < 1.0 {
        rgb = rgb.map(|value| value / alpha);
    }
    if texels.transfer == Transfer::SrgbDecoded {
        rgb = rgb.map(decode_component);
    }
    [rgb[0], rgb[1], rgb[2], alpha]
}

/// Cycles' f32 sRGB decode of one component, the `color_srgb_to_linear_sse2` path of its kernels.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Cycles decodes the interpolated value in f32, so the value is narrowed on purpose"
)]
fn decode_component(encoded: f64) -> f64 {
    f64::from(crate::color::cycles_srgb_to_linear(encoded as f32))
}

fn pixel_with_extension(
    pixels: &[[f64; 4]],
    width: usize,
    height: usize,
    x: i64,
    y: i64,
    extension: ImageExtension,
) -> [f64; 4] {
    let Some(x) = extension_index(x, width, extension) else {
        return TRANSPARENT_BLACK;
    };
    let Some(y) = extension_index(y, height, extension) else {
        return TRANSPARENT_BLACK;
    };
    pixel(pixels, width, x, y)
}

fn extension_index(index: i64, dimension: usize, extension: ImageExtension) -> Option<usize> {
    let dimension = i64::try_from(dimension).ok()?;
    let mapped = match extension {
        ImageExtension::Repeat => index.rem_euclid(dimension),
        ImageExtension::Extend => index.clamp(0, dimension - 1),
        ImageExtension::Clip => {
            if (0..dimension).contains(&index) {
                index
            } else {
                return None;
            }
        }
        ImageExtension::Mirror => {
            let period = dimension * 2;
            let index = index.rem_euclid(period);
            if index < dimension {
                index
            } else {
                period - 1 - index
            }
        }
    };
    usize::try_from(mapped).ok()
}

fn pixel(pixels: &[[f64; 4]], width: usize, x: usize, y: usize) -> [f64; 4] {
    y.checked_mul(width)
        .and_then(|offset| offset.checked_add(x))
        .and_then(|index| pixels.get(index).copied())
        .unwrap_or(TRANSPARENT_BLACK)
}

fn mix(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        ImageData, ImageExtension, ImageInterpolation, ImageStorage, ImageTileData, Precision,
        premultiply, round_to_half, sample, sample_encoded, sample_with_extension,
        sample_with_extension_alpha, sample_with_interpolation,
    };
    use std::collections::BTreeMap;

    fn data(
        width: u32,
        height: u32,
        pixels: Vec<[f64; 4]>,
        interpolation: ImageInterpolation,
    ) -> ImageData {
        ImageData {
            width,
            height,
            pixels,
            tiles: BTreeMap::new(),
            interpolation,
            srgb: false,
            storage: ImageStorage::default(),
        }
    }

    /// An sRGB two-texel row decoded to linear light, as the loader stores it.
    fn srgb_row(encoded: [f64; 2]) -> ImageData {
        ImageData {
            width: 2,
            height: 1,
            pixels: encoded
                .iter()
                .map(|&value| {
                    let linear = crate::color::srgb_to_linear(value);
                    [linear, linear, linear, 1.0]
                })
                .collect(),
            tiles: BTreeMap::new(),
            interpolation: ImageInterpolation::Linear,
            srgb: true,
            storage: ImageStorage::default(),
        }
    }

    /// Cycles' f32 decode of an encoded value, the expectation for an sRGB lookup.
    fn cycles_decode(encoded: f32) -> f64 {
        f64::from(crate::color::cycles_srgb_to_linear(encoded))
    }

    /// The two-texel 8-bit sRGB image of the blend round-trip test: pixel 0 is `[128, 40, 0, 128]` with straight alpha,
    /// pixel 1 is `[64, 0, 255, 255]`. `storage` says how the loader stored it.
    fn alpha_texels(srgb: bool, storage: ImageStorage) -> ImageData {
        let decode = |byte: f64| crate::color::srgb_to_linear(byte / 255.0);
        let pixels = if srgb {
            vec![
                [decode(128.0), decode(40.0), 0.0, 128.0 / 255.0],
                [decode(64.0), 0.0, 1.0, 1.0],
            ]
        } else {
            vec![
                [128.0 / 255.0, 40.0 / 255.0, 0.0, 128.0 / 255.0],
                [64.0 / 255.0, 0.0, 1.0, 1.0],
            ]
        };
        ImageData {
            width: 2,
            height: 1,
            pixels,
            tiles: BTreeMap::new(),
            interpolation: ImageInterpolation::Linear,
            srgb,
            storage,
        }
    }

    /// Blender 5.2.2 renders of this texel (`tests/fixtures/alpha_texture`, sRGB, Closest, Repeat, texel 0): the stored
    /// value is premultiplied at load, so an unlinked Alpha reads `decode(round(enc * alpha))`, and a linked Alpha
    /// unassociates it first. The oracle residual of this model is 6e-7 over its ramp.
    #[test]
    fn straight_alpha_srgb_texel_is_premultiplied_unless_its_alpha_is_linked() {
        let image = alpha_texels(
            true,
            ImageStorage {
                data: false,
                straight_alpha: true,
                ignore_alpha: false,
                precision: Precision::Byte,
            },
        );
        let unlinked = sample_with_extension_alpha(
            &image,
            [0.1, 0.5],
            [0.1, 0.5],
            0,
            ImageInterpolation::Closest,
            ImageExtension::Repeat,
            false,
        );
        assert!(
            (unlinked[0] - 0.051_269).abs() < 1.0e-5,
            "unlinked red is {}",
            unlinked[0]
        );
        assert!(
            (unlinked[1] - 0.006_995).abs() < 1.0e-5,
            "unlinked green is {}",
            unlinked[1]
        );
        assert!(
            (unlinked[3] - 128.0 / 255.0).abs() < 1.0e-12,
            "alpha is not premultiplied"
        );
        let linked = sample_with_extension_alpha(
            &image,
            [0.1, 0.5],
            [0.1, 0.5],
            0,
            ImageInterpolation::Closest,
            ImageExtension::Repeat,
            true,
        );
        assert!(
            (linked[0] - 0.214_041).abs() < 1.0e-5,
            "linked red is {}",
            linked[0]
        );
        assert!(
            (linked[1] - 0.021_072).abs() < 1.0e-5,
            "linked green is {}",
            linked[1]
        );
    }

    /// Blender's premultiply at load, byte precision: `floor(c * a / 255)` in integers. For c = a = 200, 200 * 200 / 255 is
    /// 156.86, so truncation gives 156 where rounding gives 157. Oracle: `tests/fixtures/alpha_texture` floor8 cases.
    #[test]
    fn byte_premultiply_truncates_like_cycles_integer_arithmetic() {
        let value = 200.0 / 255.0;
        assert!((premultiply(value, value, Precision::Byte) - 156.0 / 255.0).abs() < 1.0e-15);
        let short = 50_000.0 / 65_535.0;
        assert!(
            (premultiply(short, short, Precision::Short) - 38_147.0 / 65_535.0).abs() < 1.0e-15
        );
    }

    /// Blender's premultiply of half buffers: the product is rounded to half, ties to even. Half(0.7) * half(0.6) is
    /// 0.420185..., which rounds to 0.420166015625, not to the float product. Oracle: `tests/fixtures/alpha_texture` half cases.
    #[test]
    fn half_premultiply_rounds_the_product_to_half() {
        let colour = 0.700_195_312_5;
        let alpha = 0.600_097_656_25;
        assert!((premultiply(colour, alpha, Precision::Half) - 0.420_166_015_625).abs() < 1.0e-15);
        assert!((premultiply(colour, alpha, Precision::Float) - colour * alpha).abs() < 1.0e-15);
    }

    /// A floor texel through the sampler: pixel 0 is sRGB `[200, 100, 0]` with alpha 200. Unlinked, its red reads
    /// `decode(156 / 255)`, the truncated premultiply. Linked, it is unassociated: `156 / 200 = 0.78`, then decoded.
    #[test]
    fn floor_texel_is_truncated_then_unassociated_when_linked() {
        let decode = |byte: f64| crate::color::srgb_to_linear(byte / 255.0);
        let image = ImageData {
            width: 2,
            height: 1,
            pixels: vec![
                [decode(200.0), decode(100.0), 0.0, 200.0 / 255.0],
                [decode(64.0), 0.0, 1.0, 1.0],
            ],
            tiles: BTreeMap::new(),
            interpolation: ImageInterpolation::Linear,
            srgb: true,
            storage: ImageStorage {
                data: false,
                straight_alpha: true,
                ignore_alpha: false,
                precision: Precision::Byte,
            },
        };
        let read = |alpha_linked| {
            sample_with_extension_alpha(
                &image,
                [0.1, 0.5],
                [0.1, 0.5],
                0,
                ImageInterpolation::Closest,
                ImageExtension::Repeat,
                alpha_linked,
            )
        };
        assert!((read(false)[0] - crate::color::srgb_to_linear(156.0 / 255.0)).abs() < 1.0e-5);
        assert!((read(true)[0] - crate::color::srgb_to_linear(0.78)).abs() < 1.0e-5);
    }

    /// A Premultiplied image (Blender alpha mode PREMUL) keeps the file's values: an unlinked colour read returns them
    /// decoded, and a linked read unassociates them, `c / a` (`straight_alpha` is false). Oracle: `_premul` cases.
    #[test]
    fn premultiplied_image_keeps_its_values_and_unassociates_when_linked() {
        let image = alpha_texels(
            true,
            ImageStorage {
                data: false,
                straight_alpha: false,
                ignore_alpha: false,
                precision: Precision::Byte,
            },
        );
        let unlinked = sample_with_extension_alpha(
            &image,
            [0.1, 0.5],
            [0.1, 0.5],
            0,
            ImageInterpolation::Closest,
            ImageExtension::Repeat,
            false,
        );
        assert!((unlinked[0] - crate::color::srgb_to_linear(128.0 / 255.0)).abs() < 1.0e-5);
        let linked = sample_with_extension_alpha(
            &image,
            [0.1, 0.5],
            [0.1, 0.5],
            0,
            ImageInterpolation::Closest,
            ImageExtension::Repeat,
            true,
        );
        assert!(
            (linked[0] - 1.0).abs() < 1.0e-5,
            "linked red is {}",
            linked[0]
        );
    }

    /// Blender's `IMA_ALPHA_IGNORE` (alpha mode None): the loader makes alpha 1, no premultiply and no unassociate
    /// happen, so a linked read returns the stored colour and alpha 1. Citations: `image_metadata.cpp` (`ignore_alpha`),
    /// `shader_nodes.cpp` (IGNORE is left out of the unassociate flag).
    #[test]
    fn ignored_alpha_reads_one_and_keeps_the_stored_colour() {
        let image = alpha_texels(
            false,
            ImageStorage {
                data: false,
                straight_alpha: false,
                ignore_alpha: true,
                precision: Precision::Byte,
            },
        );
        let linked = sample_with_extension_alpha(
            &image,
            [0.1, 0.5],
            [0.1, 0.5],
            0,
            ImageInterpolation::Closest,
            ImageExtension::Repeat,
            true,
        );
        assert!(
            (linked[0] - 128.0 / 255.0).abs() < 1.0e-12,
            "stored red is {}",
            linked[0]
        );
        assert!(
            (linked[3] - 1.0).abs() < 1.0e-12,
            "ignored alpha is {}",
            linked[3]
        );
    }

    /// Round to half: a value of 1.0 is exact, and anything at or above 65520 overflows to infinity.
    #[test]
    fn round_to_half_is_ties_to_even_and_overflows_to_infinity() {
        assert!((round_to_half(1.0) - 1.0).abs() < 1.0e-15);
        assert!(round_to_half(65_520.0).is_infinite());
        assert!(round_to_half(65_504.0).is_finite());
    }

    /// A Non-Color image is read as stored, with neither a premultiply nor an unassociate, whichever Alpha link.
    #[test]
    fn non_colour_texels_are_read_as_stored_whatever_the_alpha_link() {
        let image = alpha_texels(
            false,
            ImageStorage {
                data: true,
                straight_alpha: true,
                ignore_alpha: false,
                precision: Precision::Byte,
            },
        );
        for linked in [false, true] {
            let sampled = sample_with_extension_alpha(
                &image,
                [0.1, 0.5],
                [0.1, 0.5],
                0,
                ImageInterpolation::Closest,
                ImageExtension::Repeat,
                linked,
            );
            assert!(
                (sampled[0] - 128.0 / 255.0).abs() < 1.0e-12,
                "linked {linked}: red is {}",
                sampled[0]
            );
            assert!(
                (sampled[1] - 40.0 / 255.0).abs() < 1.0e-12,
                "linked {linked}: green is {}",
                sampled[1]
            );
        }
    }

    #[test]
    fn srgb_linear_sampling_interpolates_encoded_texels_then_decodes() {
        // Cycles mixes an sRGB image's stored values and decodes the result. Halfway between encoded 0 and 1 is the
        // decoded encoded midpoint, not the midpoint of the linear values.
        let image = srgb_row([0.0, 1.0]);
        let sampled = sample_with_interpolation(&image, [0.5, 0.5], 0, ImageInterpolation::Linear);
        let expected = cycles_decode(0.5);
        assert!(
            (sampled[0] - expected).abs() < 1e-6,
            "sampled {} but Cycles' rule gives {expected}",
            sampled[0]
        );
        assert!(
            (sampled[0] - 0.5).abs() > 0.2,
            "decode-before-mix would give 0.5"
        );
        assert!((sampled[3] - 1.0).abs() < 1e-12, "alpha is not decoded");
    }

    #[test]
    fn srgb_closest_sampling_returns_the_decoded_texel() {
        let image = srgb_row([0.25, 1.0]);
        let sampled = sample_with_interpolation(&image, [0.1, 0.5], 0, ImageInterpolation::Closest);
        assert!(
            (sampled[0] - cycles_decode(0.25)).abs() < 1e-6,
            "closest reads the texel through the same decode"
        );
    }

    #[test]
    fn srgb_intensity_reads_the_encoded_texel_without_decoding() {
        // Blender's intensity channel reads the encoded bytes (`BKE_texture_get_value` with colour management off).
        let image = srgb_row([0.25, 1.0]);
        let encoded = sample_encoded(&image, [0.1, 0.5], 0);
        assert!(
            (encoded[0] - 0.25).abs() < 1e-9,
            "the encoded texel is returned"
        );
        let decoded = sample(&image, [0.1, 0.5], 0);
        assert!((decoded[0] - cycles_decode(0.25)).abs() < 1e-6);
    }

    #[test]
    fn srgb_extension_sampling_matches_the_decoded_linear_at_texel_centers() {
        let image = srgb_row([0.5, 0.5]);
        let sampled = sample_with_extension(
            &image,
            [0.25, 0.5],
            [0.25, 0.5],
            0,
            ImageInterpolation::Linear,
            ImageExtension::Extend,
        );
        assert!(
            (sampled[0] - cycles_decode(0.5)).abs() < 1e-6,
            "a texel centre is the decoded texel"
        );
    }

    #[test]
    fn linear_interpolation_uses_texel_centers_and_bilinear_weights() {
        let image = data(
            2,
            2,
            vec![
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
            ImageInterpolation::Linear,
        );

        assert_eq!(sample(&image, [0.5, 0.5], 0), [0.5, 0.5, 0.25, 0.5]);
        assert_eq!(sample(&image, [0.25, 0.25], 0), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn closest_interpolation_picks_nearest_texel() {
        let image = data(
            2,
            2,
            vec![
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            ImageInterpolation::Closest,
        );

        assert_eq!(sample(&image, [0.74, 0.26], 0), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(sample(&image, [1.0, 1.0], 0), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn explicit_interpolation_overrides_shared_image_default() {
        let image = data(
            2,
            1,
            vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            ImageInterpolation::Closest,
        );

        assert_eq!(sample(&image, [0.5, 0.5], 0), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            sample_with_interpolation(&image, [0.5, 0.5], 0, ImageInterpolation::Linear),
            [0.5, 0.5, 0.0, 1.0],
        );
    }

    #[test]
    fn extension_sampler_wraps_bilinear_seams_for_repeat_and_mirror() {
        let image = data(
            2,
            1,
            vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            ImageInterpolation::Linear,
        );

        assert_eq!(
            sample_with_extension(
                &image,
                [0.0, 0.5],
                [0.0, 0.5],
                0,
                ImageInterpolation::Linear,
                ImageExtension::Repeat,
            ),
            [0.5, 0.5, 0.0, 1.0],
        );
        assert_eq!(
            sample_with_extension(
                &image,
                [0.0, 0.5],
                [0.0, 0.5],
                0,
                ImageInterpolation::Linear,
                ImageExtension::Mirror,
            ),
            [1.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn extension_sampler_clamps_extend_and_returns_black_outside_clip() {
        let image = data(
            2,
            1,
            vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            ImageInterpolation::Linear,
        );

        assert_eq!(
            sample_with_extension(
                &image,
                [1.5, 0.5],
                [1.5, 0.5],
                0,
                ImageInterpolation::Linear,
                ImageExtension::Extend,
            ),
            [0.0, 1.0, 0.0, 1.0],
        );
        assert_eq!(
            sample_with_extension(
                &image,
                [1.01, 0.5],
                [1.01, 0.5],
                0,
                ImageInterpolation::Linear,
                ImageExtension::Clip,
            ),
            [0.0; 4],
        );
    }

    #[test]
    fn clip_linear_sampling_uses_transparent_black_border_texels() {
        let image = data(
            2,
            1,
            vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            ImageInterpolation::Linear,
        );

        assert_eq!(
            sample_with_extension(
                &image,
                [0.0, 0.5],
                [0.0, 0.5],
                0,
                ImageInterpolation::Linear,
                ImageExtension::Clip,
            ),
            [0.5, 0.0, 0.0, 0.5],
        );
    }

    #[test]
    fn clip_closest_at_one_selects_the_last_texel() {
        let image = data(
            2,
            1,
            vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            ImageInterpolation::Closest,
        );

        assert_eq!(
            sample_with_extension(
                &image,
                [1.0, 1.0],
                [1.0, 1.0],
                0,
                ImageInterpolation::Closest,
                ImageExtension::Clip,
            ),
            [0.0, 1.0, 0.0, 1.0],
        );
    }

    #[test]
    fn native_sampler_keeps_clamped_linear_edge_behavior() {
        let image = data(
            2,
            1,
            vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            ImageInterpolation::Linear,
        );

        assert_eq!(
            sample_with_interpolation(&image, [0.0, 0.5], 0, ImageInterpolation::Linear),
            [1.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn udim_lookup_applies_uv_offsets_and_falls_back_to_base_tile() {
        let mut image = data(
            1,
            1,
            vec![[0.25, 0.5, 0.75, 1.0]],
            ImageInterpolation::Closest,
        );
        image.tiles.insert(
            1012,
            ImageTileData {
                width: 2,
                height: 1,
                pixels: vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
                srgb: false,
            },
        );

        assert_eq!(sample(&image, [1.25, 1.25], 1001), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(sample(&image, [0.25, 0.25], 1001), [0.25, 0.5, 0.75, 1.0]);
        assert_eq!(sample(&image, [1.25, 0.25], 1001), [0.0; 4]);
        assert_eq!(
            sample_with_extension(
                &image,
                [1.0, 1.5],
                [0.0, 0.5],
                1001,
                ImageInterpolation::Linear,
                ImageExtension::Repeat,
            ),
            [0.5, 0.5, 0.0, 1.0],
        );
    }

    #[test]
    fn malformed_images_and_nonfinite_uvs_return_transparent_black() {
        assert_eq!(
            sample(
                &data(0, 1, vec![], ImageInterpolation::Linear),
                [0.5, 0.5],
                0,
            ),
            [0.0; 4]
        );
        assert_eq!(
            sample(
                &data(2, 2, vec![[1.0; 4]; 3], ImageInterpolation::Closest),
                [0.5, 0.5],
                0,
            ),
            [0.0; 4]
        );
        assert_eq!(
            sample(
                &data(1, 1, vec![[1.0; 4]], ImageInterpolation::Linear),
                [f64::NAN, 0.5],
                0,
            ),
            [0.0; 4]
        );
    }
}
