use serde::{Deserialize, Serialize};
use std::{borrow::Cow, collections::BTreeMap, sync::OnceLock};

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

/// One UDIM tile. `pixels` are linear light, as stored. `precision` is the precision of the tile's own file, which sets
/// the rounding of its premultiply (see [`Precision`]). The tile's colour space and storage flags are the base image's.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageTileData {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f64; 4]>,
    pub precision: Precision,
    /// The tile's stored colour texels for colour reads (see [`ColourCache`]).
    pub colour: ColourCache,
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
    /// The stored colour texels for colour reads, computed on the first colour read (see [`ColourCache`]).
    pub colour: ColourCache,
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

/// The stored colour texels an Image Texture colour read mixes: each texel encoded for an sRGB image and premultiplied at
/// the image's precision, as [`stored_texel`] makes them. They depend only on the pixels and the storage, so they are
/// computed once, on the first colour read, and every later sample reuses them. This is the form Cycles' loaded buffer
/// holds them in (`util_image_multiply_native`, `NODE_IMAGE_COMPRESS_AS_SRGB`).
#[derive(Clone, Debug, Default)]
pub struct ColourCache(OnceLock<Vec<[f64; 4]>>);

impl PartialEq for ColourCache {
    /// The cache is derived from the pixels and the storage, so it never makes two images differ.
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

/// The stored colour texels of `pixels` for a colour read of an image with these settings, computed once into `cache`.
fn colour_texels<'a>(
    cache: &'a ColourCache,
    pixels: &[[f64; 4]],
    srgb: bool,
    storage: ImageStorage,
) -> &'a [[f64; 4]] {
    cache.0.get_or_init(|| {
        let texels = Texels::new(
            srgb,
            storage,
            Read::Colour {
                alpha_linked: false,
            },
        );
        pixels
            .iter()
            .map(|pixel| stored_texel(*pixel, texels))
            .collect()
    })
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
    /// The values are already stored, encoded and premultiplied in a [`ColourCache`], so [`stored_texel`] returns them.
    precomputed: bool,
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
            precomputed: false,
        }
    }
}

/// Samples an image at a normalized UV coordinate, with the colour rule of Blender's texture modifiers: an sRGB
/// image's encoded texels are mixed, and the result is decoded to linear light. Blender's `BKE_texture_get_value`
/// reads the straight encoded bytes (`ibuf_get_color`), mixes them in `boxsample` (a one-texel box, which is the
/// bilinear weighting), and then converts the colour channels with `IMB_colormanagement_colorspace_to_scene_linear_v3`.
/// The modifier reader does not premultiply.
///
/// With `udim` set, the tile is the one the UV's floored offsets select from tile 1001; a missing tile reads transparent
/// black.
#[must_use]
pub fn sample(data: &ImageData, uv: [f64; 2], udim: bool) -> [f64; 4] {
    sample_image(data, uv, udim, data.interpolation, Read::Decoded)
}

/// Samples an image's straight stored values as Blender's intensity channel reads them: an sRGB image's encoded texels
/// are mixed and returned encoded, with no decode. Other images return the same values as [`sample`].
#[must_use]
pub fn sample_encoded(data: &ImageData, uv: [f64; 2], udim: bool) -> [f64; 4] {
    sample_image(data, uv, udim, data.interpolation, Read::Encoded)
}

fn sample_image(
    data: &ImageData,
    uv: [f64; 2],
    udim: bool,
    interpolation: ImageInterpolation,
    read: Read,
) -> [f64; 4] {
    if !uv[0].is_finite() || !uv[1].is_finite() {
        return TRANSPARENT_BLACK;
    }

    let Some(image) = image_for_uv(data, uv, udim) else {
        return TRANSPARENT_BLACK;
    };
    let Some((width, height)) = image.valid_dimensions() else {
        return TRANSPARENT_BLACK;
    };
    let local_uv = if udim {
        [uv[0] - uv[0].floor(), uv[1] - uv[1].floor()]
    } else {
        uv
    };
    let u = local_uv[0].clamp(0.0, 1.0);
    let v = local_uv[1].clamp(0.0, 1.0);

    let (values, texels) = image.source(read);
    match interpolation {
        ImageInterpolation::Linear => sample_linear(values, width, height, u, v, texels),
        ImageInterpolation::Closest => sample_closest(values, width, height, u, v, texels),
    }
}

/// The colour of a UDIM lookup whose tile is missing or outside the tile grid: Cycles' `IMAGE_MISSING_RGBA`, which
/// Blender shows as magenta (the `blender_udim_*` cases in `tests/fixtures/repeat_edge`).
const MISSING_TILE_RGBA: [f64; 4] = [1.0, 0.0, 1.0, 1.0];

/// Samples an image as an Image Texture node reads it. `uv` is Blender's V-up coordinate. A UDIM image (`udim`) routes its
/// tile from `uv` ([`udim_route`]), and the lookup inside it is the same as an image's, at the tile-local coordinate
/// (Cycles samples each UDIM tile as its own image: `kernel_image_udim_map`, then the interpolators of
/// `kernel/device/cpu/image.h`). A missing tile is [`MISSING_TILE_RGBA`]. `alpha_linked` is whether the node's Alpha
/// output is linked, which is when Cycles unassociates the colour (see [`ImageStorage`]).
#[must_use]
pub fn sample_with_extension_alpha(
    data: &ImageData,
    uv: [f64; 2],
    udim: bool,
    interpolation: ImageInterpolation,
    extension: ImageExtension,
    alpha_linked: bool,
) -> [f64; 4] {
    if !uv[0].is_finite() || !uv[1].is_finite() {
        return TRANSPARENT_BLACK;
    }
    let (image, local) = if udim {
        let Some((number, local)) = udim_route(uv) else {
            return MISSING_TILE_RGBA;
        };
        let Some(image) = stored_tile(data, number) else {
            return MISSING_TILE_RGBA;
        };
        (image, local)
    } else {
        (base_tile(data), uv)
    };
    let Some((width, height)) = image.valid_dimensions() else {
        return TRANSPARENT_BLACK;
    };
    let (values, texels) = image.source(Read::Colour { alpha_linked });
    match interpolation {
        // The linear lookup reads Potter's V-down storage: the flipped coordinate, with the extension on storage indices.
        ImageInterpolation::Linear => sample_linear_with_extension(
            values,
            width,
            height,
            local[0],
            1.0 - local[1],
            extension,
            texels,
        ),
        ImageInterpolation::Closest => {
            sample_closest_cycles(values, width, height, local, extension, texels)
        }
    }
}

/// Cycles' UDIM routing (`kernel_image_udim_map`): the tile holding `uv` and the coordinate relative to it. Each coordinate
/// is truncated, so a coordinate in (-1, 0) belongs to tile column or row 0. A tile column of 10 or more, a negative
/// column, or a negative row has no tile.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the tile number is checked against the u32 range before the conversion, and it is never negative"
)]
fn udim_route(uv: [f64; 2]) -> Option<(u32, [f64; 2])> {
    let (column, row) = (uv[0].trunc(), uv[1].trunc());
    if !(0.0..10.0).contains(&column) || row < 0.0 {
        return None;
    }
    let number = 1001.0 + 10.0 * row + column;
    if number > f64::from(u32::MAX) {
        return None;
    }
    Some((number as u32, [uv[0] - column, uv[1] - row]))
}

fn base_tile(data: &ImageData) -> ImageTileDataRef<'_> {
    ImageTileDataRef {
        width: data.width,
        height: data.height,
        pixels: &data.pixels,
        colour: &data.colour,
        srgb: data.srgb,
        storage: data.storage,
    }
}

/// The stored tile `number`. The base image is tile 1001, which Potter keeps outside `tiles`. Any other tile must be stored.
fn stored_tile(data: &ImageData, number: u32) -> Option<ImageTileDataRef<'_>> {
    match data.tiles.get(&number) {
        Some(image) => Some(ImageTileDataRef {
            width: image.width,
            height: image.height,
            pixels: &image.pixels,
            colour: &image.colour,
            srgb: data.srgb,
            storage: ImageStorage {
                precision: image.precision,
                ..data.storage
            },
        }),
        None if number == 1001 => Some(base_tile(data)),
        None => None,
    }
}

/// The image a floor-routed lookup reads (`sample`, `sample_encoded`): the base image when `udim` is unset, else the stored
/// tile the UV's floored offsets select. A missing tile has no pixels, so the lookup returns transparent black.
fn image_for_uv(data: &ImageData, uv: [f64; 2], udim: bool) -> Option<ImageTileDataRef<'_>> {
    if !udim {
        return Some(base_tile(data));
    }
    stored_tile(data, udim_tile(uv)?)
}

struct ImageTileDataRef<'a> {
    width: u32,
    height: u32,
    pixels: &'a [[f64; 4]],
    colour: &'a ColourCache,
    srgb: bool,
    storage: ImageStorage,
}

impl<'a> ImageTileDataRef<'a> {
    fn valid_dimensions(&self) -> Option<(usize, usize)> {
        if self.width == 0 || self.height == 0 {
            return None;
        }
        let width = usize::try_from(self.width).ok()?;
        let height = usize::try_from(self.height).ok()?;
        let pixel_count = width.checked_mul(height)?;
        (self.pixels.len() == pixel_count).then_some((width, height))
    }

    /// The values a read mixes, and the settings it stores them with. A colour read of an encoded or premultiplied image
    /// mixes the cached colour texels; every other read mixes the stored pixels and stores each texel as it reads it.
    fn source(&self, read: Read) -> (&'a [[f64; 4]], Texels) {
        let texels = Texels::new(self.srgb, self.storage, read);
        if matches!(read, Read::Colour { .. })
            && (texels.transfer != Transfer::Linear || texels.premultiply.is_some())
        {
            let values = colour_texels(self.colour, self.pixels, self.srgb, self.storage);
            return (
                values,
                Texels {
                    precomputed: true,
                    ..texels
                },
            );
        }
        (self.pixels, texels)
    }
}

const I64_UPPER_EXCLUSIVE: f64 = 9_223_372_036_854_775_808.0;
const I64_LOWER_INCLUSIVE: f64 = -9_223_372_036_854_775_808.0;
/// The UDIM tile at the floored UV offsets from tile 1001: the floor routing `sample` and `sample_encoded` use.
fn udim_tile(uv: [f64; 2]) -> Option<u32> {
    let u_offset = floor_to_i64(uv[0])?;
    let v_offset = floor_to_i64(uv[1])?;
    let offset = v_offset.checked_mul(10)?.checked_add(u_offset)?;
    let actual_tile = 1001_i64.checked_add(offset)?;
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

/// Cycles' `frac` index of a coordinate (`kernel/device/cpu/image.h`): the integer part, truncated, and one less when the
/// coordinate is negative. That is `floor` for every coordinate except an exact negative integer, which Cycles reads one
/// texel lower (`floor(x) - 1`), so `-12.0` gives `-13`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the truncated input is checked against exclusive i64 conversion bounds"
)]
fn cycles_frac_index(value: f64) -> Option<i64> {
    let truncated = value.trunc();
    if !(I64_LOWER_INCLUSIVE..I64_UPPER_EXCLUSIVE).contains(&truncated) {
        return None;
    }
    let index = truncated as i64;
    if value < 0.0 {
        index.checked_sub(1)
    } else {
        Some(index)
    }
}

/// The closest texel of a full-image lookup, as Cycles' `interp_closest` reads it (`kernel/device/cpu/image.h`). The texel
/// is Cycles' `frac` of the Blender coordinate scaled by the image size, with no offset, then the extension maps it.
/// Blender's rows run bottom-up and the storage runs top-down, so the storage row is the mirror of the wrapped row. For a
/// non-integer coordinate this is the floor of the coordinate, as before. At an exact integer the top-down flip used to
/// pick the neighbouring texel, and at an exact negative integer Cycles reads one texel lower.
#[expect(
    clippy::cast_precision_loss,
    reason = "validated u32 image dimensions are exactly representable as f64"
)]
fn sample_closest_cycles(
    pixels: &[[f64; 4]],
    width: usize,
    height: usize,
    uv: [f64; 2],
    extension: ImageExtension,
    texels: Texels,
) -> [f64; 4] {
    let (Some(column), Some(row)) = (
        cycles_frac_index(uv[0] * width as f64),
        cycles_frac_index(uv[1] * height as f64),
    ) else {
        return TRANSPARENT_BLACK;
    };
    let (Some(x), Some(y)) = (
        extension_index(column, width, extension),
        extension_index(row, height, extension),
    ) else {
        return TRANSPARENT_BLACK;
    };
    finish(
        stored_texel(pixel(pixels, width, x, height - 1 - y), texels),
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
    if texels.precomputed {
        return linear;
    }
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
/// Magnitudes above the largest finite half clamp to 65504, as `float_to_half_image` does (`intern/cycles/util/half.h`).
fn round_to_half(value: f64) -> f64 {
    if value.is_nan() || value.abs() < f64::MIN_POSITIVE {
        return value;
    }
    let magnitude = value.abs().min(65_504.0);
    let exponent = i32::try_from((magnitude.to_bits() >> 52) & 0x7ff).unwrap_or(0) - 1023;
    let step = 2.0_f64.powi(exponent.max(-14) - 10);
    value.signum() * ((magnitude / step).round_ties_even() * step)
}

/// The colour a lookup returns from a mixed stored value. Cycles unassociates the colour when the Alpha output is
/// linked (`svm_image_texture`, `NODE_IMAGE_ALPHA_UNASSOCIATE`, for any alpha but 0 and 1), and then
/// decodes an sRGB image to linear light (`color_srgb_to_linear_v4`). Alpha is unchanged.
fn finish(mixed: [f64; 4], texels: Texels) -> [f64; 4] {
    // Alpha ignored (`IMA_ALPHA_IGNORE`): the loader made alpha 1, so every read returns 1 (`image_metadata.cpp`).
    let alpha = if texels.ignore_alpha { 1.0 } else { mixed[3] };
    let mut rgb = [mixed[0], mixed[1], mixed[2]];
    // Cycles unassociates for any alpha but 0 and 1 (`alpha != 1.0f && alpha != 0.0f`): outside [0, 1], or strictly inside.
    let unassociates = alpha < 0.0 || alpha > 1.0 || (alpha > 0.0 && alpha < 1.0);
    if texels.unassociate && unassociates {
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

/// Fixture grids shared by the sampler and world tests, read from `tests/fixtures/repeat_edge`.
#[cfg(test)]
pub(crate) mod test_fixtures {
    /// The texels of a fixture grid (`texels_blender_rows`, bottom-up rows of `width` texels) in the storage order Potter
    /// keeps (top-down rows).
    pub(crate) fn fixture_pixels(
        width: usize,
        height: usize,
        blender_rows: &[[f64; 4]],
    ) -> Vec<[f64; 4]> {
        let mut pixels = vec![[0.0; 4]; width * height];
        for row in 0..height {
            for x in 0..width {
                pixels[row * width + x] = blender_rows[(height - 1 - row) * width + x];
            }
        }
        pixels
    }

    pub(crate) fn fixture_texels(
        value: &serde_json::Value,
    ) -> Result<Vec<[f64; 4]>, Box<dyn std::error::Error>> {
        value
            .as_array()
            .ok_or("texels are missing")?
            .iter()
            .map(|texel| -> Result<[f64; 4], Box<dyn std::error::Error>> {
                let channel = |index: usize| texel[index].as_f64().ok_or("texel is not a number");
                Ok([channel(0)?, channel(1)?, channel(2)?, channel(3)?])
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::{fixture_pixels, fixture_texels};
    use super::{
        ColourCache, ImageData, ImageExtension, ImageInterpolation, ImageStorage, ImageTileData,
        Precision, premultiply, round_to_half, sample, sample_encoded, sample_with_extension_alpha,
        udim_route,
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
            colour: ColourCache::default(),
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
            colour: ColourCache::default(),
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
            colour: ColourCache::default(),
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
            false,
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
            false,
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
            colour: ColourCache::default(),
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
                false,
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
            false,
            ImageInterpolation::Closest,
            ImageExtension::Repeat,
            false,
        );
        assert!((unlinked[0] - crate::color::srgb_to_linear(128.0 / 255.0)).abs() < 1.0e-5);
        let linked = sample_with_extension_alpha(
            &image,
            [0.1, 0.5],
            false,
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
            false,
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

    /// Round to half: a value of 1.0 is exact, and a magnitude above the largest half (65504) clamps to it, as Cycles'
    /// `float_to_half_image` does.
    #[test]
    fn round_to_half_is_ties_to_even_and_clamps_to_the_largest_half() {
        assert!((round_to_half(1.0) - 1.0).abs() < 1.0e-15);
        assert!((round_to_half(65_520.0) - 65_504.0).abs() < 1.0e-9);
        assert!((round_to_half(-1.0e6) + 65_504.0).abs() < 1.0e-9);
        assert!((round_to_half(65_504.0) - 65_504.0).abs() < 1.0e-9);
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
                false,
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
        let sampled = sample_with_extension_alpha(
            &image,
            [0.5, 0.5],
            false,
            ImageInterpolation::Linear,
            ImageExtension::Repeat,
            false,
        );
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
        let sampled = sample_with_extension_alpha(
            &image,
            [0.1, 0.5],
            false,
            ImageInterpolation::Closest,
            ImageExtension::Repeat,
            false,
        );
        assert!(
            (sampled[0] - cycles_decode(0.25)).abs() < 1e-6,
            "closest reads the texel through the same decode"
        );
    }

    #[test]
    fn srgb_intensity_reads_the_encoded_texel_without_decoding() {
        // Blender's intensity channel reads the encoded bytes (`BKE_texture_get_value` with colour management off).
        let image = srgb_row([0.25, 1.0]);
        let encoded = sample_encoded(&image, [0.1, 0.5], false);
        assert!(
            (encoded[0] - 0.25).abs() < 1e-9,
            "the encoded texel is returned"
        );
        let decoded = sample(&image, [0.1, 0.5], false);
        assert!((decoded[0] - cycles_decode(0.25)).abs() < 1e-6);
    }

    #[test]
    fn srgb_extension_sampling_matches_the_decoded_linear_at_texel_centers() {
        let image = srgb_row([0.5, 0.5]);
        let sampled = sample_with_extension_alpha(
            &image,
            [0.25, 0.5],
            false,
            ImageInterpolation::Linear,
            ImageExtension::Extend,
            false,
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

        assert_eq!(sample(&image, [0.5, 0.5], false), [0.5, 0.5, 0.25, 0.5]);
        assert_eq!(sample(&image, [0.25, 0.25], false), [0.0, 0.0, 0.0, 0.0]);
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

        assert_eq!(sample(&image, [0.74, 0.26], false), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(sample(&image, [1.0, 1.0], false), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn explicit_interpolation_overrides_shared_image_default() {
        let image = data(
            2,
            1,
            vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            ImageInterpolation::Closest,
        );

        assert_eq!(sample(&image, [0.5, 0.5], false), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            sample_with_extension_alpha(
                &image,
                [0.5, 0.5],
                false,
                ImageInterpolation::Linear,
                ImageExtension::Repeat,
                false,
            ),
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
            sample_with_extension_alpha(
                &image,
                [0.0, 0.5],
                false,
                ImageInterpolation::Linear,
                ImageExtension::Repeat,
                false,
            ),
            [0.5, 0.5, 0.0, 1.0],
        );
        assert_eq!(
            sample_with_extension_alpha(
                &image,
                [0.0, 0.5],
                false,
                ImageInterpolation::Linear,
                ImageExtension::Mirror,
                false,
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
            sample_with_extension_alpha(
                &image,
                [1.5, 0.5],
                false,
                ImageInterpolation::Linear,
                ImageExtension::Extend,
                false,
            ),
            [0.0, 1.0, 0.0, 1.0],
        );
        assert_eq!(
            sample_with_extension_alpha(
                &image,
                [1.25, 0.5],
                false,
                ImageInterpolation::Linear,
                ImageExtension::Clip,
                false,
            ),
            [0.0; 4],
        );
        // A linear sample partly inside a clip keeps its inside texel's weight (per texel, as Blender's clip does): at
        // u = 0.875 the coordinate is 1.25 texels, so texel 1 weighs 0.75 and the clipped texel 2 contributes nothing.
        assert_eq!(
            sample_with_extension_alpha(
                &image,
                [0.875, 0.5],
                false,
                ImageInterpolation::Linear,
                ImageExtension::Clip,
                false,
            ),
            [0.0, 0.75, 0.0, 0.75],
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
            sample_with_extension_alpha(
                &image,
                [0.0, 0.5],
                false,
                ImageInterpolation::Linear,
                ImageExtension::Clip,
                false,
            ),
            [0.5, 0.0, 0.0, 0.5],
        );
    }

    #[test]
    fn clip_closest_at_one_is_outside_the_image_as_cycles_reads_it() {
        let image = data(
            2,
            1,
            vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            ImageInterpolation::Closest,
        );

        assert_eq!(
            sample_with_extension_alpha(
                &image,
                [1.0, 1.0],
                false,
                ImageInterpolation::Closest,
                ImageExtension::Clip,
                false,
            ),
            [0.0; 4],
        );
    }

    #[test]
    fn world_sampler_wraps_the_seam_as_cycles_environment_textures_do() {
        let image = data(
            2,
            1,
            vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            ImageInterpolation::Linear,
        );

        assert_eq!(
            sample_with_extension_alpha(
                &image,
                [0.0, 0.5],
                false,
                ImageInterpolation::Linear,
                ImageExtension::Repeat,
                false,
            ),
            [0.5, 0.5, 0.0, 1.0],
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
                precision: Precision::Float,
                colour: ColourCache::default(),
            },
        );

        assert_eq!(sample(&image, [1.25, 1.25], true), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(sample(&image, [0.25, 0.25], true), [0.25, 0.5, 0.75, 1.0]);
        assert_eq!(sample(&image, [1.25, 0.25], true), [0.0; 4]);
        assert_eq!(
            sample_with_extension_alpha(
                &image,
                [1.0, 1.5],
                true,
                ImageInterpolation::Linear,
                ImageExtension::Repeat,
                false,
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
                false,
            ),
            [0.0; 4]
        );
        assert_eq!(
            sample(
                &data(2, 2, vec![[1.0; 4]; 3], ImageInterpolation::Closest),
                [0.5, 0.5],
                false,
            ),
            [0.0; 4]
        );
        assert_eq!(
            sample(
                &data(1, 1, vec![[1.0; 4]], ImageInterpolation::Linear),
                [f64::NAN, 0.5],
                false,
            ),
            [0.0; 4]
        );
    }

    /// The 8-bit texels of the generator's base table (`generate_alpha_oracle.py`, `INTEGER_TEXELS`).
    const BASE8: [u8; 8] = [128, 40, 0, 128, 64, 0, 255, 255];
    /// The 8-bit texels of the `floor` table, where the premultiply truncates.
    const FLOOR8: [u8; 8] = [200, 100, 0, 200, 64, 0, 255, 255];
    /// The 16-bit texels of the `floor` table.
    const FLOOR16: [u16; 8] = [50_000, 20_000, 0, 50_000, 20_000, 0, 65_535, 65_535];
    /// The half-float texels (`HALF_TEXELS`), rounded to half when the EXR is read.
    const HALF_TEXELS: [f64; 8] = [0.7, 0.35, 0.0, 0.6, 0.3, 0.0, 1.0, 1.0];
    /// The float texels of the single-layer 32-bit EXR cases (`FLOAT_TEXELS`).
    const FLOAT_TEXELS: [f64; 8] = [0.8, 0.4, 0.2, 0.5, 0.3, 0.0, 1.0, 1.0];
    /// The associated texels Potter's EXR writer produces (`POTTER_TEXELS`).
    const POTTER_TEXELS: [f64; 8] = [0.4, 0.2, 0.1, 0.5, 0.15, 0.0, 0.5, 0.5];

    /// The stored values of one oracle family, in the file's units, with the precision and straight flag Blender loads
    /// it with. The suffix after `<colourspace>_<interpolation>_<extension>_<linked>` names the family.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "float EXR channels are f32, so the table values are rounded to f32 on purpose"
    )]
    fn oracle_family(suffix: &str) -> Option<(Vec<f64>, Precision, bool)> {
        let byte = |table: [u8; 8]| table.map(|value| f64::from(value) / 255.0).to_vec();
        let short = |table: [u16; 8]| table.map(|value| f64::from(value) / 65_535.0).to_vec();
        let float = |table: [f64; 8]| table.map(|value| f64::from(value as f32)).to_vec();
        let half = |table: [f64; 8]| table.map(round_to_half).to_vec();
        match suffix {
            "" | "base8_linear" => Some((byte(BASE8), Precision::Byte, true)),
            "base8_premul" => Some((byte(BASE8), Precision::Byte, false)),
            "floor8_straight" | "floor8_tiff" => Some((byte(FLOOR8), Precision::Byte, true)),
            "png16_straight" | "floor16_tiff16" => Some((short(FLOOR16), Precision::Short, true)),
            "half16_exr" => Some((half(HALF_TEXELS), Precision::Half, true)),
            "float_straight_exr32" => Some((float(FLOAT_TEXELS), Precision::Float, true)),
            "float_premul_exr32" => Some((float(FLOAT_TEXELS), Precision::Float, false)),
            "potter_premul_exr32" => Some((float(POTTER_TEXELS), Precision::Float, false)),
            _ => None,
        }
    }

    /// Rebuilds one oracle case's image from its file name (`<colourspace>_<interpolation>_<extension>_<linked>[_<family>]`
    /// or `multilayer_<interpolation>_<extension>_<linked>`) and returns it with the settings Blender rendered it with.
    /// A multilayer case's texels are the ones Blender loaded from its EXR, which the JSON stores.
    fn oracle_case(
        stem: &str,
        document: &serde_json::Value,
    ) -> Result<(ImageData, ImageInterpolation, ImageExtension, bool), Box<dyn std::error::Error>>
    {
        let (colourspace, rest) = if let Some(rest) = stem.strip_prefix("srgb_") {
            ("sRGB", rest)
        } else if let Some(rest) = stem.strip_prefix("linear_") {
            ("Linear", rest)
        } else if let Some(rest) = stem.strip_prefix("non-color_") {
            ("Non-Color", rest)
        } else if let Some(rest) = stem.strip_prefix("multilayer_") {
            ("Linear", rest)
        } else {
            return Err(format!("unclassified oracle case {stem}").into());
        };
        let mut parts = rest.splitn(4, '_');
        let interpolation = match parts.next() {
            Some("closest") => ImageInterpolation::Closest,
            Some("linear") => ImageInterpolation::Linear,
            _ => return Err(format!("unclassified interpolation in {stem}").into()),
        };
        let extension = match parts.next() {
            Some("repeat") => ImageExtension::Repeat,
            Some("extend") => ImageExtension::Extend,
            _ => return Err(format!("unclassified extension in {stem}").into()),
        };
        let linked = match parts.next() {
            Some("linked") => true,
            Some("unlinked") => false,
            _ => return Err(format!("unclassified alpha link in {stem}").into()),
        };
        let suffix = parts.next().unwrap_or("");
        let srgb = colourspace == "sRGB";
        let (width, height, pixels, precision, straight) = if stem.starts_with("multilayer_") {
            let texels = document["texels"]
                .as_array()
                .ok_or("multilayer texels are missing")?;
            let pixels = texels
                .iter()
                .map(|texel| -> Result<[f64; 4], Box<dyn std::error::Error>> {
                    let channel =
                        |index: usize| texel[index].as_f64().ok_or("texel channel is not a number");
                    Ok([channel(0)?, channel(1)?, channel(2)?, channel(3)?])
                })
                .collect::<Result<Vec<_>, _>>()?;
            let width = u32::try_from(document["width"].as_u64().ok_or("width is missing")?)?;
            let height = u32::try_from(document["height"].as_u64().ok_or("height is missing")?)?;
            (width, height, pixels, Precision::Float, false)
        } else {
            let Some((file, precision, straight)) = oracle_family(suffix) else {
                return Err(format!("unclassified texel family in {stem}").into());
            };
            let pixels: Vec<[f64; 4]> = file
                .as_chunks::<4>()
                .0
                .iter()
                .map(|texel| {
                    let colour = |value: f64| {
                        if srgb {
                            crate::color::srgb_to_linear(value)
                        } else {
                            value
                        }
                    };
                    [
                        colour(texel[0]),
                        colour(texel[1]),
                        colour(texel[2]),
                        texel[3],
                    ]
                })
                .collect();
            (2, 1, pixels, precision, straight)
        };
        let image = ImageData {
            width,
            height,
            pixels,
            tiles: BTreeMap::new(),
            interpolation,
            srgb,
            storage: ImageStorage {
                data: colourspace == "Non-Color",
                straight_alpha: straight,
                ignore_alpha: false,
                precision,
            },
            colour: ColourCache::default(),
        };
        Ok((image, interpolation, extension, linked))
    }

    /// Every committed Blender oracle case (`tests/fixtures/alpha_texture`, 60 files) samples to Blender 5.2.2's render
    /// colour at each of its 1024 records, within 2e-6. The generator's fit leaves at most 8.4e-7 over these cases, and
    /// each record is rounded to six decimals (5e-7), so 2e-6 is that residual plus f32 rounding, not a loosened check.
    #[test]
    fn alpha_texture_samples_match_blenders_render() -> Result<(), Box<dyn std::error::Error>> {
        const TOLERANCE: f64 = 2.0e-6;
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/alpha_texture");
        let mut cases = 0_usize;
        for entry in std::fs::read_dir(&directory)? {
            let path = entry?.path();
            let Some(file) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some(stem) = file
                .strip_prefix("blender_")
                .and_then(|rest| rest.strip_suffix(".json"))
            else {
                continue;
            };
            let document: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
            let (image, interpolation, extension, linked) = oracle_case(stem, &document)?;
            let records = document["records"]
                .as_array()
                .ok_or("records are missing")?;
            let mut worst = 0.0_f64;
            for record in records {
                let value =
                    |index: usize| record[index].as_f64().ok_or("record value is not a number");
                let (sample_u, sample_v) = (value(5)?, value(6)?);
                let expected = [value(2)?, value(3)?, value(4)?];
                // The records give Blender's UV (V up); the sampler flips V into the stored rows.
                let actual = sample_with_extension_alpha(
                    &image,
                    [sample_u, sample_v],
                    false,
                    interpolation,
                    extension,
                    linked,
                );
                for channel in 0..3 {
                    worst = worst.max((actual[channel] - expected[channel]).abs());
                }
            }
            assert!(
                worst <= TOLERANCE,
                "{stem}: worst residual {worst:e} exceeds {TOLERANCE:e}"
            );
            cases += 1;
        }
        assert_eq!(cases, 60, "every committed oracle case must be checked");
        Ok(())
    }

    fn non_colour_storage() -> ImageStorage {
        ImageStorage {
            data: true,
            straight_alpha: false,
            ignore_alpha: false,
            precision: Precision::Float,
        }
    }

    /// Blender 5.2.2 at exact and near-edge UVs (`tests/fixtures/repeat_edge`). The eight `blender_edge_*` cases are one
    /// 2 x 2 texture with 16 x 16 pixel-centre UVs in eighths from -0.75 to 1.125, for each of the four extensions and
    /// both interpolations. The eight `blender_udim_*` cases are UDIM tiles 1001 and 1002 with UVs from 0.25 to 2.125, so
    /// tile 1002 is entered at exactly u = 1.0, and tile 1003 is missing (magenta). Each record is sampled at the Blender
    /// UV the Image Texture received, and every colour must match to 1e-6 (the fixtures round to six decimals).
    #[test]
    fn edge_oracles_match_blenders_render() -> Result<(), Box<dyn std::error::Error>> {
        const TOLERANCE: f64 = 1.0e-6;
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/repeat_edge");
        let mut cases = 0_usize;
        for entry in std::fs::read_dir(&directory)? {
            let path = entry?.path();
            let Some(file) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if path.extension().and_then(|extension| extension.to_str()) != Some("json")
                || !(file.starts_with("blender_edge_") || file.starts_with("blender_udim_"))
            {
                continue;
            }
            let document: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
            let interpolation = match document["interpolation"].as_str() {
                Some("Closest") => ImageInterpolation::Closest,
                Some("Linear") => ImageInterpolation::Linear,
                _ => return Err(format!("{file}: interpolation is missing").into()),
            };
            let extension = match document["extension"].as_str() {
                Some("REPEAT") => ImageExtension::Repeat,
                Some("EXTEND") => ImageExtension::Extend,
                Some("CLIP") => ImageExtension::Clip,
                Some("MIRROR") => ImageExtension::Mirror,
                _ => return Err(format!("{file}: extension is missing").into()),
            };
            let (image, udim) = if document["kind"].as_str() == Some("udim") {
                let base = &document["tiles"]["1001"];
                let width = usize::try_from(base["width"].as_u64().ok_or("width is missing")?)?;
                let pixels =
                    fixture_pixels(width, width, &fixture_texels(&base["texels_blender_rows"])?);
                let mut tiles = BTreeMap::new();
                let second = &document["tiles"]["1002"];
                let second_pixels = fixture_pixels(
                    width,
                    width,
                    &fixture_texels(&second["texels_blender_rows"])?,
                );
                tiles.insert(
                    1002,
                    ImageTileData {
                        width: u32::try_from(width)?,
                        height: u32::try_from(width)?,
                        pixels: second_pixels,
                        precision: Precision::Float,
                        colour: ColourCache::default(),
                    },
                );
                let image = ImageData {
                    width: u32::try_from(width)?,
                    height: u32::try_from(width)?,
                    pixels,
                    tiles,
                    interpolation,
                    srgb: false,
                    storage: non_colour_storage(),
                    colour: ColourCache::default(),
                };
                (image, true)
            } else {
                let width = usize::try_from(document["width"].as_u64().ok_or("width is missing")?)?;
                let pixels = fixture_pixels(
                    width,
                    width,
                    &fixture_texels(&document["texels_blender_rows"])?,
                );
                let image = ImageData {
                    width: u32::try_from(width)?,
                    height: u32::try_from(width)?,
                    pixels,
                    tiles: BTreeMap::new(),
                    interpolation,
                    srgb: false,
                    storage: non_colour_storage(),
                    colour: ColourCache::default(),
                };
                (image, false)
            };
            let mut worst = 0.0_f64;
            for record in document["records"]
                .as_array()
                .ok_or("records are missing")?
            {
                let value =
                    |index: usize| record[index].as_f64().ok_or("record value is not a number");
                let uv = [value(5)?, value(6)?];
                let expected = [value(2)?, value(3)?, value(4)?];
                let actual =
                    sample_with_extension_alpha(&image, uv, udim, interpolation, extension, false);
                for channel in 0..3 {
                    worst = worst.max((actual[channel] - expected[channel]).abs());
                }
            }
            assert!(
                worst <= TOLERANCE,
                "{file}: worst residual {worst:e} exceeds {TOLERANCE:e}"
            );
            cases += 1;
        }
        assert_eq!(
            cases, 16,
            "every committed edge oracle case must be checked"
        );
        Ok(())
    }

    #[test]
    fn udim_routing_truncates_each_coordinate_as_cycles_does() {
        // A coordinate in (-1, 0) stays in column or row 0 and is relative to it, so it is not missing.
        assert_eq!(udim_route([-0.25, 0.5]), Some((1001, [-0.25, 0.5])));
        assert_eq!(udim_route([1.0, -0.25]), Some((1002, [0.0, -0.25])));
        assert_eq!(udim_route([1.5, 1.25]), Some((1012, [0.5, 0.25])));
        // Columns outside 0..10 and negative rows have no tile (`kernel_image_udim_map`).
        assert_eq!(udim_route([-1.5, 0.5]), None);
        assert_eq!(udim_route([10.0, 0.5]), None);
        assert_eq!(udim_route([0.5, -1.0]), None);
    }
}
