use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    fs,
    io::Cursor,
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::{
    error::{ErrorCode, PotError, Result},
    hash,
    image::{ColourCache, ImageData, ImageInterpolation, ImageStorage, ImageTileData, Precision},
    model::{Image, ImageAlphaMode, ImageColorspace},
};

const PIXEL_FORMAT: &str = "potter.rgba_f64.v1";
pub const MAX_IMAGE_PIXELS: u64 = 16_777_216;

/// Static layer/pass/view selectors; still frames are fixed and tiled textures derive tiles from UVs.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ImageUserSelection {
    pub(crate) layer: u16,
    pub(crate) pass: u16,
    pub(crate) view: u16,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredPixels {
    format: String,
    width: u32,
    height: u32,
    pixels: Vec<[f64; 4]>,
}
#[derive(Serialize)]
struct StoredPixelsRef<'a> {
    format: &'static str,
    width: u32,
    height: u32,
    pixels: &'a [[f64; 4]],
}

/// Encodes generated or edited linear-RGBA pixels into the immutable asset format.
///
/// # Errors
/// Returns an error when dimensions do not match the pixel count or serialization fails.
pub fn encode_pixels(width: u32, height: u32, pixels: &[[f64; 4]]) -> Result<Vec<u8>> {
    validate_pixels(width, height, pixels)?;
    serde_json::to_vec(&StoredPixelsRef {
        format: PIXEL_FORMAT,
        width,
        height,
        pixels,
    })
    .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))
}

/// Reads an image's base pixels and UDIM tiles from its scene asset store.
/// Linked images are checked against their registered source hash before use.
///
/// # Errors
/// Returns an error for a missing or changed asset, unsupported image encoding, or invalid pixels.
pub fn load_image_data(
    image: &Image,
    root: &Path,
    interpolation: ImageInterpolation,
) -> Result<ImageData> {
    load_image_data_with_staged(image, root, interpolation, &BTreeMap::new())
}

/// As [`load_image_data`], but also resolves immutable blobs staged by the active apply batch.
pub fn load_image_data_with_staged(
    image: &Image,
    root: &Path,
    interpolation: ImageInterpolation,
    staged: &BTreeMap<String, Vec<u8>>,
) -> Result<ImageData> {
    load_image_data_with_staged_and_user(image, root, interpolation, staged, None)
        .map(|(image, _, _)| image)
}

/// Loads an image using the selection captured by a legacy Image Texture's `ImageUser`.
/// The legacy sampler reads straight texels, so a premultiplied EXR RGBA colour pass is
/// unpremultiplied here; other passes keep their stored values.
pub(crate) fn load_image_data_for_legacy_texture(
    image: &Image,
    root: &Path,
    interpolation: ImageInterpolation,
    selection: ImageUserSelection,
) -> Result<(ImageData, bool, usize)> {
    load_image_data_with_staged_and_user(
        image,
        root,
        interpolation,
        &BTreeMap::new(),
        Some(selection),
    )
}

fn load_image_data_with_staged_and_user(
    image: &Image,
    root: &Path,
    interpolation: ImageInterpolation,
    staged: &BTreeMap<String, Vec<u8>>,
    selection: Option<ImageUserSelection>,
) -> Result<(ImageData, bool, usize)> {
    let decode = |bytes: &[u8]| {
        if let Some(selection) = selection {
            decode_pixels_with_image_user(bytes, image.colorspace, image.alpha_mode, selection)
        } else {
            decode_pixels(bytes, image.colorspace)
                .map(|(width, height, pixels)| (width, height, pixels, true, 4))
        }
    };
    let srgb = decodes_srgb(image.colorspace);
    // A UDIM image whose base file is missing keeps its listed tiles and no base pixels, so the base samples as missing.
    let (width, height, pixels, color_pass, source_channels, storage) = if image.base_is_missing() {
        (0, 0, Vec::new(), true, 4, ImageStorage::default())
    } else {
        let base_bytes = image_blob_bytes(image.blob.as_deref(), image, root, staged)?;
        let (width, height, pixels, color_pass, source_channels) = decode(&base_bytes)?;
        if width != image.width || height != image.height {
            return Err(PotError::with_details(
                ErrorCode::SceneInvalid,
                "image blob dimensions do not match its registry entry",
                serde_json::json!({"image":image.name,"expected":[image.width,image.height],"actual":[width,height]}),
            ));
        }
        let mut storage = image_storage(image, &base_bytes);
        // The legacy decode unpremultiplies a premultiplied EXR colour pass, so its texels are straight
        // and the storage says so.
        if selection.is_some()
            && base_bytes.starts_with(&[0x76, 0x2f, 0x31, 0x01])
            && legacy_colour_is_straight(image.alpha_mode, color_pass, source_channels)
        {
            storage.straight_alpha = true;
        }
        (width, height, pixels, color_pass, source_channels, storage)
    };
    let mut tiles = BTreeMap::new();
    for tile in &image.tiles {
        // A listed tile without a blob is missing: it has no pixels, so it is left out and samples as the missing colour.
        let Some(blob) = tile.blob.as_deref() else {
            continue;
        };
        let bytes = blob_bytes(blob, root, staged)?;
        let (tile_width, tile_height, tile_pixels, _, _) = decode(&bytes)?;
        if tile_width != tile.width || tile_height != tile.height {
            return Err(PotError::with_details(
                ErrorCode::SceneInvalid,
                "UDIM blob dimensions do not match its registry entry",
                serde_json::json!({"tile":tile.number,"expected":[tile.width,tile.height],"actual":[tile_width,tile_height]}),
            ));
        }
        tiles.insert(
            tile.number,
            ImageTileData {
                width: tile_width,
                height: tile_height,
                pixels: tile_pixels,
                precision: source_precision(&bytes),
                colour: ColourCache::default(),
            },
        );
    }
    Ok((
        ImageData {
            width,
            height,
            pixels,
            tiles,
            interpolation,
            srgb,
            storage,
            udim: image.is_udim(),
            colour: ColourCache::default(),
        },
        color_pass,
        source_channels,
    ))
}

type DecodedImageUserPixels = (u32, u32, Vec<[f64; 4]>, bool, usize);

fn decode_pixels_with_image_user(
    bytes: &[u8],
    colorspace: ImageColorspace,
    alpha_mode: ImageAlphaMode,
    selection: ImageUserSelection,
) -> Result<DecodedImageUserPixels> {
    if bytes.starts_with(&[0x76, 0x2f, 0x31, 0x01]) {
        let (width, height, mut pixels, color_pass, source_channels) =
            decode_exr_with_image_user(bytes, selection, alpha_mode)?;
        if color_pass {
            pixels = apply_colorspace(pixels, colorspace);
            if source_channels == 1 {
                for pixel in &mut pixels {
                    pixel[3] = pixel[0];
                }
            }
        }
        return Ok((width, height, pixels, color_pass, source_channels));
    }
    let (width, height, pixels) = decode_pixels(bytes, colorspace)?;
    Ok((width, height, pixels, true, 4))
}

/// Whether an image's stored values are sRGB-encoded: an sRGB colour space on any source. Cycles keeps such an image
/// encoded and decodes after it interpolates (`ImageMetaData::is_compressible_as_srgb`). Potter's stored pixel format
/// returns its values as stored, so the label alone decides here, as it does for the `.blend` and glTF exports that
/// encode an sRGB-labelled image before Blender reads it.
pub(crate) fn decodes_srgb(colorspace: ImageColorspace) -> bool {
    colorspace == ImageColorspace::Srgb
}

/// The alpha mode Blender gives a loaded image of this encoding when the file says nothing else. Blender reads an EXR
/// with the `OpenEXR` convention, associated (premultiplied) alpha, because `oiio:UnassociatedAlpha` is absent: a loaded
/// EXR reports `alpha_mode` PREMUL (`image_metadata.cpp` `ImageMetaData`, and the `alpha_texture` readback oracle).
/// Every other encoding is straight.
pub(crate) fn default_alpha_mode(bytes: &[u8]) -> ImageAlphaMode {
    if bytes.starts_with(&[0x76, 0x2f, 0x31, 0x01]) {
        ImageAlphaMode::Premultiplied
    } else {
        ImageAlphaMode::Straight
    }
}

/// How Blender stores a loaded image for Image Texture colour reads: a non-colour image is read as stored, and
/// straight alpha is premultiplied at load for any other colour space (Cycles' image manager).
fn image_storage(image: &Image, bytes: &[u8]) -> ImageStorage {
    ImageStorage {
        data: image.colorspace == ImageColorspace::NonColor,
        straight_alpha: image.alpha_mode == ImageAlphaMode::Straight,
        // Blender's `IMA_ALPHA_IGNORE` is the None alpha mode: the alpha of the file is ignored (`image_metadata.cpp`).
        ignore_alpha: image.alpha_mode == ImageAlphaMode::None,
        precision: source_precision(bytes),
    }
}

/// The precision a stored image is quantised to, which sets the precision of Blender's premultiply at load: a 16-bit
/// PNG or any 16-bit decoder output is 16-bit; an EXR whose colour channels are all half is half; an EXR with float
/// channels, Radiance HDR, float decoder output, and Potter's stored pixel format are float; other encodings are 8-bit.
fn source_precision(bytes: &[u8]) -> Precision {
    if bytes.first() == Some(&b'{') {
        return Precision::Float;
    }
    if bytes.starts_with(&[0x76, 0x2f, 0x31, 0x01]) {
        return exr_precision(bytes);
    }
    if bytes.starts_with(b"#?") {
        return Precision::Float;
    }
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return if bytes.get(24) == Some(&16) {
            Precision::Short
        } else {
            Precision::Byte
        };
    }
    decoded_precision(bytes)
}

/// Half when every colour channel of the EXR is F16, float otherwise (`exr::meta::attribute::SampleType`).
fn exr_precision(bytes: &[u8]) -> Precision {
    let Ok(metadata) = exr::meta::MetaData::read_from_buffered(Cursor::new(bytes), false) else {
        return Precision::Float;
    };
    let colour_types: Vec<exr::meta::attribute::SampleType> = metadata
        .headers
        .iter()
        .flat_map(|header| header.channels.list.iter())
        .filter(|channel| {
            let name = channel.name.to_string().to_ascii_lowercase();
            let suffix = name.rsplit('.').next().unwrap_or("");
            matches!(
                suffix,
                "r" | "g" | "b" | "a" | "red" | "green" | "blue" | "alpha"
            )
        })
        .map(|channel| channel.sample_type)
        .collect();
    if !colour_types.is_empty()
        && colour_types
            .iter()
            .all(|sample| *sample == exr::meta::attribute::SampleType::F16)
    {
        Precision::Half
    } else {
        Precision::Float
    }
}

/// The precision of a decoder's colour output, for formats the image crate decodes (TIFF, JPEG, and others).
fn decoded_precision(bytes: &[u8]) -> Precision {
    use ::image::ImageDecoder as _;
    let color_type = ::image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()
        .and_then(|reader| reader.into_decoder().ok())
        .map(|decoder| decoder.color_type());
    match color_type {
        Some(::image::ColorType::Rgb32F | ::image::ColorType::Rgba32F) => Precision::Float,
        Some(
            ::image::ColorType::L16
            | ::image::ColorType::La16
            | ::image::ColorType::Rgb16
            | ::image::ColorType::Rgba16,
        ) => Precision::Short,
        _ => Precision::Byte,
    }
}

/// Decodes supported image encodings into linear RGBA pixels.
///
/// Eight-bit rasters default to sRGB; EXR, Radiance HDR, and floating-point TIFF default
/// to linear. EXR layers named `Combined` or `RGBA` are preferred over the first RGBA
/// layer. EXR values are returned as stored, straight or premultiplied as the image's alpha mode says; the image
/// sampler applies that (`ImageStorage`), and `ImageData::straight_pixels` gives straight colour for direct readers.
pub fn decode_pixels(
    bytes: &[u8],
    colorspace: ImageColorspace,
) -> Result<(u32, u32, Vec<[f64; 4]>)> {
    let (_, width, height, pixels) =
        decode_pixels_with_default_colorspace(bytes, Some(colorspace))?;
    Ok((width, height, pixels))
}

pub(crate) fn decode_pixels_with_default_colorspace(
    bytes: &[u8],
    colorspace: Option<ImageColorspace>,
) -> Result<(ImageColorspace, u32, u32, Vec<[f64; 4]>)> {
    if let Ok(stored) = serde_json::from_slice::<StoredPixels>(bytes)
        && stored.format == PIXEL_FORMAT
    {
        validate_pixels(stored.width, stored.height, &stored.pixels)?;
        return Ok((
            colorspace.unwrap_or(ImageColorspace::Linear),
            stored.width,
            stored.height,
            stored.pixels,
        ));
    }
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        let selected_colorspace = colorspace.unwrap_or(ImageColorspace::Srgb);
        let (width, height, pixels) = decode_png(bytes, selected_colorspace)?;
        return Ok((selected_colorspace, width, height, pixels));
    }
    if bytes.starts_with(&[0x76, 0x2f, 0x31, 0x01]) {
        let (width, height, pixels) = decode_exr(bytes)?;
        let selected_colorspace = colorspace.unwrap_or(ImageColorspace::Linear);
        let pixels = apply_colorspace(pixels, selected_colorspace);
        return Ok((selected_colorspace, width, height, pixels));
    }

    let format = match ::image::guess_format(bytes) {
        Ok(format) => format,
        Err(_) if looks_like_tga(bytes) => ::image::ImageFormat::Tga,
        Err(error) => {
            return Err(PotError::with_details(
                ErrorCode::InvalidArgument,
                format!("image encoding could not be identified: {error}"),
                serde_json::json!({"feature_id":"image.encoding.unknown.invalid_data"}),
            ));
        }
    };
    let format_name = image_format_name(format);
    if !matches!(
        format,
        ::image::ImageFormat::Jpeg
            | ::image::ImageFormat::Tiff
            | ::image::ImageFormat::WebP
            | ::image::ImageFormat::Bmp
            | ::image::ImageFormat::Tga
            | ::image::ImageFormat::Hdr
    ) {
        return Err(encoding_error(
            ErrorCode::UnsupportedFeature,
            format_name,
            "unsupported_variant",
            "image format is not enabled for decoding",
        ));
    }
    decode_raster(bytes, format, format_name, colorspace)
}

fn image_format_name(format: ::image::ImageFormat) -> &'static str {
    match format {
        ::image::ImageFormat::Jpeg => "jpeg",
        ::image::ImageFormat::Tiff => "tiff",
        ::image::ImageFormat::WebP => "webp",
        ::image::ImageFormat::Bmp => "bmp",
        ::image::ImageFormat::Tga => "tga",
        ::image::ImageFormat::Hdr => "hdr",
        ::image::ImageFormat::Png => "png",
        _ => "unknown",
    }
}

fn looks_like_tga(bytes: &[u8]) -> bool {
    let Some(header) = bytes.get(..18) else {
        return false;
    };
    let image_type = header[2];
    let color_map_type = header[1];
    let width = u16::from_le_bytes([header[12], header[13]]);
    let height = u16::from_le_bytes([header[14], header[15]]);
    matches!(color_map_type, 0 | 1)
        && matches!(image_type, 1 | 2 | 3 | 9 | 10 | 11)
        && width > 0
        && height > 0
        && matches!(header[16], 8 | 15 | 16 | 24 | 32)
}

fn decode_raster(
    bytes: &[u8],
    format: ::image::ImageFormat,
    format_name: &str,
    colorspace: Option<ImageColorspace>,
) -> Result<(ImageColorspace, u32, u32, Vec<[f64; 4]>)> {
    use ::image::ImageDecoder;

    let decoder = ::image::ImageReader::with_format(Cursor::new(bytes), format)
        .into_decoder()
        .map_err(|error| raster_error(format_name, &error))?;
    let (width, height) = decoder.dimensions();
    checked_pixel_count(width, height)?;
    let image = ::image::DynamicImage::from_decoder(decoder)
        .map_err(|error| raster_error(format_name, &error))?;
    let is_float = matches!(
        &image,
        ::image::DynamicImage::ImageRgb32F(_) | ::image::DynamicImage::ImageRgba32F(_)
    );
    let selected_colorspace = colorspace.unwrap_or(if is_float {
        ImageColorspace::Linear
    } else {
        ImageColorspace::Srgb
    });
    let pixels = match image {
        ::image::DynamicImage::ImageLuma8(image) => convert_samples(
            width,
            height,
            image.pixels().map(|pixel| {
                let gray = f64::from(pixel[0]) / 255.0;
                [gray, gray, gray, 1.0]
            }),
            selected_colorspace,
        )?,
        ::image::DynamicImage::ImageLumaA8(image) => convert_samples(
            width,
            height,
            image.pixels().map(|pixel| {
                let gray = f64::from(pixel[0]) / 255.0;
                [gray, gray, gray, f64::from(pixel[1]) / 255.0]
            }),
            selected_colorspace,
        )?,
        ::image::DynamicImage::ImageRgb8(image) => convert_samples(
            width,
            height,
            image.pixels().map(|pixel| {
                [
                    f64::from(pixel[0]) / 255.0,
                    f64::from(pixel[1]) / 255.0,
                    f64::from(pixel[2]) / 255.0,
                    1.0,
                ]
            }),
            selected_colorspace,
        )?,
        ::image::DynamicImage::ImageRgba8(image) => convert_samples(
            width,
            height,
            image.pixels().map(|pixel| {
                [
                    f64::from(pixel[0]) / 255.0,
                    f64::from(pixel[1]) / 255.0,
                    f64::from(pixel[2]) / 255.0,
                    f64::from(pixel[3]) / 255.0,
                ]
            }),
            selected_colorspace,
        )?,
        ::image::DynamicImage::ImageLuma16(image) => convert_samples(
            width,
            height,
            image.pixels().map(|pixel| {
                let gray = f64::from(pixel[0]) / 65_535.0;
                [gray, gray, gray, 1.0]
            }),
            selected_colorspace,
        )?,
        ::image::DynamicImage::ImageLumaA16(image) => convert_samples(
            width,
            height,
            image.pixels().map(|pixel| {
                let gray = f64::from(pixel[0]) / 65_535.0;
                [gray, gray, gray, f64::from(pixel[1]) / 65_535.0]
            }),
            selected_colorspace,
        )?,
        ::image::DynamicImage::ImageRgb16(image) => convert_samples(
            width,
            height,
            image.pixels().map(|pixel| {
                [
                    f64::from(pixel[0]) / 65_535.0,
                    f64::from(pixel[1]) / 65_535.0,
                    f64::from(pixel[2]) / 65_535.0,
                    1.0,
                ]
            }),
            selected_colorspace,
        )?,
        ::image::DynamicImage::ImageRgba16(image) => convert_samples(
            width,
            height,
            image.pixels().map(|pixel| {
                [
                    f64::from(pixel[0]) / 65_535.0,
                    f64::from(pixel[1]) / 65_535.0,
                    f64::from(pixel[2]) / 65_535.0,
                    f64::from(pixel[3]) / 65_535.0,
                ]
            }),
            selected_colorspace,
        )?,
        ::image::DynamicImage::ImageRgb32F(image) => convert_samples(
            width,
            height,
            image.pixels().map(|pixel| {
                [
                    f64::from(pixel[0]),
                    f64::from(pixel[1]),
                    f64::from(pixel[2]),
                    1.0,
                ]
            }),
            selected_colorspace,
        )?,
        ::image::DynamicImage::ImageRgba32F(image) => convert_samples(
            width,
            height,
            image.pixels().map(|pixel| {
                [
                    f64::from(pixel[0]),
                    f64::from(pixel[1]),
                    f64::from(pixel[2]),
                    f64::from(pixel[3]),
                ]
            }),
            selected_colorspace,
        )?,
        _ => {
            return Err(encoding_error(
                ErrorCode::UnsupportedFeature,
                format_name,
                "unsupported_pixel_type",
                "decoded image pixel type is unsupported",
            ));
        }
    };
    Ok((selected_colorspace, width, height, pixels))
}

fn convert_samples(
    width: u32,
    height: u32,
    samples: impl Iterator<Item = [f64; 4]>,
    colorspace: ImageColorspace,
) -> Result<Vec<[f64; 4]>> {
    let pixels = apply_colorspace(samples.collect(), colorspace);
    validate_pixels(width, height, &pixels)?;
    Ok(pixels)
}

fn apply_colorspace(mut pixels: Vec<[f64; 4]>, colorspace: ImageColorspace) -> Vec<[f64; 4]> {
    if colorspace == ImageColorspace::Srgb {
        for pixel in &mut pixels {
            for component in &mut pixel[..3] {
                *component = crate::color::srgb_to_linear(*component);
            }
        }
    }
    pixels
}

fn raster_error(format: &str, error: &::image::ImageError) -> PotError {
    let message = error.to_string();
    let lowercase = message.to_ascii_lowercase();
    let details_variant = if lowercase.contains("cmyk") {
        "cmyk"
    } else if lowercase.contains("deep") {
        "deep_data"
    } else {
        "unsupported_variant"
    };
    let code = match error {
        ::image::ImageError::Unsupported(_) => ErrorCode::UnsupportedFeature,
        ::image::ImageError::Limits(_) => ErrorCode::LimitExceeded,
        _ => ErrorCode::InvalidArgument,
    };
    encoding_error(code, format, details_variant, &message)
}

fn encoding_error(code: ErrorCode, format: &str, variant: &str, message: &str) -> PotError {
    PotError::with_details(
        code,
        format!("{format} image decoding failed: {message}"),
        serde_json::json!({"feature_id":format!("image.encoding.{format}.{variant}")}),
    )
}

type ExrChannelData = exr::image::AnyChannel<exr::image::FlatSamples>;
type ExrLayerData = exr::image::Layer<exr::image::AnyChannels<exr::image::FlatSamples>>;
type ExrImageData = exr::image::FlatImage;

fn read_exr_data(bytes: &[u8]) -> Result<ExrImageData> {
    use exr::prelude::{ReadChannels, ReadLayers};

    let metadata =
        exr::meta::MetaData::read_from_buffered(Cursor::new(bytes), false).map_err(|error| {
            encoding_error(
                ErrorCode::InvalidArgument,
                "exr",
                "invalid_data",
                &error.to_string(),
            )
        })?;
    for header in &metadata.headers {
        if header.deep {
            return Err(encoding_error(
                ErrorCode::UnsupportedFeature,
                "exr",
                "deep_data",
                "deep EXR images are unsupported",
            ));
        }
        let width = u32::try_from(header.layer_size.width()).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "EXR width exceeds platform limits",
            )
        })?;
        let height = u32::try_from(header.layer_size.height()).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "EXR height exceeds platform limits",
            )
        })?;
        checked_pixel_count(width, height)?;
    }

    exr::prelude::read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .all_layers()
        .all_attributes()
        .from_buffered(Cursor::new(bytes))
        .map_err(|error| {
            let message = error.to_string();
            let lower = message.to_ascii_lowercase();
            let (code, variant) = if lower.contains("deep") {
                (ErrorCode::UnsupportedFeature, "deep_data")
            } else {
                (ErrorCode::InvalidArgument, "invalid_data")
            };
            encoding_error(code, "exr", variant, &message)
        })
}

/// Decodes an Ocean bake EXR as stored. Blender writes these without finite-value validation, so
/// the samples are kept as they are.
pub(crate) fn decode_ocean_cache_exr(bytes: &[u8]) -> Result<(u32, u32, Vec<[f64; 4]>)> {
    decode_exr_default_with_validation(&read_exr_data(bytes)?, false)
}

fn decode_exr(bytes: &[u8]) -> Result<(u32, u32, Vec<[f64; 4]>)> {
    let (width, height, pixels) = decode_exr_default(&read_exr_data(bytes)?)?;
    Ok((width, height, pixels))
}

fn decode_exr_with_image_user(
    bytes: &[u8],
    selection: ImageUserSelection,
    alpha_mode: ImageAlphaMode,
) -> Result<DecodedImageUserPixels> {
    let image = read_exr_data(bytes)?;
    let multilayer = exr_is_multilayer(&image);
    if !multilayer {
        let (width, height, mut pixels) = decode_exr_default(&image)?;
        if legacy_colour_is_straight(alpha_mode, true, 4) {
            for pixel in &mut pixels {
                unpremultiply_exr_pixel(pixel);
            }
        }
        return Ok((width, height, pixels, true, 4));
    }

    let views = exr_views(&image);
    let layers = exr_render_layers(&image, &views);
    let first_layer = image.layer_data.first().ok_or_else(|| {
        encoding_error(
            ErrorCode::UnsupportedFeature,
            "exr",
            "no_layers",
            "EXR contains no image layers",
        )
    })?;
    let width = u32::try_from(first_layer.size.width()).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EXR width exceeds platform limits",
        )
    })?;
    let height = u32::try_from(first_layer.size.height()).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EXR height exceeds platform limits",
        )
    })?;
    let Some(pass) = select_exr_render_pass(&layers, selection) else {
        let pixel_count = checked_pixel_count(width, height)?;
        return Ok((width, height, vec![[0.0; 4]; pixel_count], false, 0));
    };
    let width = u32::try_from(pass.width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EXR width exceeds platform limits",
        )
    })?;
    let height = u32::try_from(pass.height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EXR height exceeds platform limits",
        )
    })?;
    let pixel_count = checked_pixel_count(width, height)?;

    let (channel_slots, channel_ids) = exr_pass_channel_layout(pass);
    let channel_count = pass.channels.len();
    let color_pass = is_exr_color_pass(channel_count, &channel_ids);
    let straight_colour = legacy_colour_is_straight(alpha_mode, color_pass, channel_count);
    let mut pixels = Vec::with_capacity(pixel_count);
    for index in 0..pixel_count {
        let mut pixel = if channel_count == 1 {
            let value = exr_sample(pass.channels[0].channel, index)?;
            [value; 4]
        } else if channel_count == 3 || channel_count == 4 {
            let mut values = [0.0; 4];
            for (slot, channel_index) in channel_slots.iter().enumerate() {
                if let Some(channel_index) = channel_index {
                    values[slot] = exr_sample(pass.channels[*channel_index].channel, index)?;
                }
            }
            if channel_count == 3 {
                values[3] = 1.0;
            }
            values
        } else {
            let value = exr_sample(pass.channels[0].channel, index)?;
            [value; 4]
        };
        if straight_colour {
            unpremultiply_exr_pixel(&mut pixel);
        }
        pixels.push(pixel);
    }
    validate_pixels(width, height, &pixels)?;
    Ok((width, height, pixels, color_pass, channel_count))
}

/// Decodes the first RGB layer with its values as stored (see [`ImageData::pixels`]).
fn decode_exr_default(image: &ExrImageData) -> Result<(u32, u32, Vec<[f64; 4]>)> {
    decode_exr_default_with_validation(image, true)
}

fn decode_exr_default_with_validation(
    image: &ExrImageData,
    validate_finite: bool,
) -> Result<(u32, u32, Vec<[f64; 4]>)> {
    let has_rgb = |layer: &&ExrLayerData| {
        exr_channel(layer, &["r", "red"]).is_some()
            && exr_channel(layer, &["g", "green"]).is_some()
            && exr_channel(layer, &["b", "blue"]).is_some()
    };
    let layer = image
        .layer_data
        .iter()
        .filter(has_rgb)
        .min_by_key(|layer| {
            let name = layer
                .attributes
                .layer_name
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default();
            match name
                .rsplit('.')
                .next()
                .unwrap_or("")
                .to_ascii_lowercase()
                .as_str()
            {
                "combined" => 0,
                "rgba" => 1,
                _ => 2,
            }
        })
        .ok_or_else(|| {
            encoding_error(
                ErrorCode::UnsupportedFeature,
                "exr",
                "no_rgba_layer",
                "EXR contains no layer with RGB channels",
            )
        })?;
    let width = u32::try_from(layer.size.width()).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EXR width exceeds platform limits",
        )
    })?;
    let height = u32::try_from(layer.size.height()).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EXR height exceeds platform limits",
        )
    })?;
    let pixel_count = checked_pixel_count(width, height)?;
    let red = exr_channel(layer, &["r", "red"]).ok_or_else(|| {
        encoding_error(
            ErrorCode::UnsupportedFeature,
            "exr",
            "no_rgba_layer",
            "EXR red channel is missing",
        )
    })?;
    let green = exr_channel(layer, &["g", "green"]).ok_or_else(|| {
        encoding_error(
            ErrorCode::UnsupportedFeature,
            "exr",
            "no_rgba_layer",
            "EXR green channel is missing",
        )
    })?;
    let blue = exr_channel(layer, &["b", "blue"]).ok_or_else(|| {
        encoding_error(
            ErrorCode::UnsupportedFeature,
            "exr",
            "no_rgba_layer",
            "EXR blue channel is missing",
        )
    })?;
    let alpha = exr_channel(layer, &["a", "alpha"]);
    let mut pixels = Vec::with_capacity(pixel_count);
    for index in 0..pixel_count {
        // The stored values, as Blender reads them: the alpha mode of the image says whether they are straight or
        // premultiplied, so no unpremultiply happens here (`ImageData::straight_pixels` does it for direct readers).
        pixels.push([
            exr_sample(red, index)?,
            exr_sample(green, index)?,
            exr_sample(blue, index)?,
            alpha.map_or(Ok(1.0), |channel| exr_sample(channel, index))?,
        ]);
    }
    if validate_finite {
        validate_pixels(width, height, &pixels)?;
    }
    Ok((width, height, pixels))
}

/// Whether a legacy image's decoded colour pass is straight. A premultiplied RGBA colour pass is
/// unpremultiplied at decode, as F's `image_user` decode did, so the legacy sampler reads straight
/// texels. Z, depth, and other passes keep their stored values, and so do the other alpha modes.
fn legacy_colour_is_straight(
    alpha_mode: ImageAlphaMode,
    color_pass: bool,
    channel_count: usize,
) -> bool {
    alpha_mode == ImageAlphaMode::Premultiplied && color_pass && channel_count == 4
}

fn unpremultiply_exr_pixel(pixel: &mut [f64; 4]) {
    let alpha = pixel[3];
    if alpha == 0.0 {
        pixel[..3].fill(0.0);
    } else {
        for component in &mut pixel[..3] {
            *component /= alpha;
        }
    }
}

fn exr_channel<'a>(
    layer: &'a exr::image::Layer<exr::image::AnyChannels<exr::image::FlatSamples>>,
    names: &[&str],
) -> Option<&'a exr::image::AnyChannel<exr::image::FlatSamples>> {
    layer.channel_data.list.iter().find(|channel| {
        let name = channel.name.to_string().to_ascii_lowercase();
        let channel = name.rsplit('.').next().unwrap_or("");
        names
            .iter()
            .any(|candidate| name == *candidate || channel == *candidate)
    })
}

fn exr_sample(
    channel: &exr::image::AnyChannel<exr::image::FlatSamples>,
    index: usize,
) -> Result<f64> {
    let value = match &channel.sample_data {
        exr::image::FlatSamples::F16(samples) => {
            samples.get(index).map(|sample| f64::from(sample.to_f32()))
        }
        exr::image::FlatSamples::F32(samples) => {
            samples.get(index).map(|sample| f64::from(*sample))
        }
        exr::image::FlatSamples::U32(_) => {
            return Err(encoding_error(
                ErrorCode::UnsupportedFeature,
                "exr",
                "uint_channels",
                "unsigned-integer EXR channels are unsupported",
            ));
        }
    };
    value.ok_or_else(|| {
        encoding_error(
            ErrorCode::UnsupportedFeature,
            "exr",
            "subsampled_channels",
            "EXR channel sample dimensions do not match the image",
        )
    })
}

struct ExrSelectionChannel<'a> {
    channel: &'a ExrChannelData,
    id: char,
}

struct ExrSelectionPass<'a> {
    name: String,
    view: String,
    view_id: usize,
    width: usize,
    height: usize,
    channels: Vec<ExrSelectionChannel<'a>>,
}

struct ExrSelectionLayer<'a> {
    passes: Vec<ExrSelectionPass<'a>>,
}

struct ExrRawChannel<'a> {
    channel: &'a ExrChannelData,
    name: String,
    view: String,
    width: usize,
    height: usize,
}

fn exr_is_multilayer(image: &ExrImageData) -> bool {
    image.layer_data.len() > 1
        || image.layer_data.iter().any(|layer| {
            layer
                .attributes
                .multi_view_names
                .as_ref()
                .is_some_and(|views| !views.is_empty())
                || layer
                    .channel_data
                    .list
                    .iter()
                    .any(|channel| channel.name.to_string().contains('.'))
        })
}

fn exr_views(image: &ExrImageData) -> Vec<String> {
    let mut views = Vec::new();
    for layer in &image.layer_data {
        if let Some(view) = &layer.attributes.view_name {
            push_exr_view(&mut views, &view.to_string());
        }
        if let Some(layer_views) = &layer.attributes.multi_view_names {
            for view in layer_views {
                push_exr_view(&mut views, &view.to_string());
            }
        }
    }
    if views.is_empty() {
        views.push(String::new());
    }
    let mut ordered = Vec::with_capacity(views.len());
    for stereo_view in ["left", "right"] {
        if let Some(index) = views.iter().position(|view| view == stereo_view) {
            ordered.push(views.remove(index));
        }
    }
    ordered.extend(views);
    ordered
}

fn push_exr_view(views: &mut Vec<String>, view: &str) {
    if !views.iter().any(|known| known == view) {
        views.push(view.to_owned());
    }
}

fn exr_render_layers<'a>(image: &'a ExrImageData, views: &[String]) -> Vec<ExrSelectionLayer<'a>> {
    let mut raw_channels = Vec::new();
    for layer in &image.layer_data {
        let layer_views = layer
            .attributes
            .multi_view_names
            .as_ref()
            .filter(|views| !views.is_empty())
            .map(|views| views.iter().map(ToString::to_string).collect::<Vec<_>>());
        let mut part_name = layer
            .attributes
            .layer_name
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default();
        let part_view = layer
            .attributes
            .view_name
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default();
        if layer_views.is_none()
            && !part_view.is_empty()
            && let Some(prefix) = part_name
                .strip_suffix(&part_view)
                .and_then(|name| name.strip_suffix('.'))
                .or_else(|| {
                    part_name
                        .strip_suffix(&part_view)
                        .and_then(|name| name.strip_suffix('-'))
                })
        {
            part_name = prefix.to_owned();
        }

        for channel in &layer.channel_data.list {
            let stored_name = channel.name.to_string();
            let (mut name, view) = if let Some(layer_views) = &layer_views {
                remove_exr_view_name(&stored_name, layer_views)
            } else {
                (stored_name, part_view.clone())
            };
            if !part_name.is_empty()
                && !name
                    .strip_prefix(&part_name)
                    .is_some_and(|suffix| suffix.starts_with('.'))
            {
                let mut full_name = String::with_capacity(part_name.len() + name.len() + 1);
                full_name.push_str(&part_name);
                full_name.push('.');
                full_name.push_str(&name);
                name = full_name;
            }
            raw_channels.push(ExrRawChannel {
                channel,
                name,
                view,
                width: layer.size.width(),
                height: layer.size.height(),
            });
        }
    }

    let has_xyz_channels = raw_channels
        .iter()
        .any(|channel| matches!(channel.name.as_str(), "X" | "Y" | "Z"));
    let mut layer_indices = HashMap::<String, usize>::new();
    let mut layers = Vec::<ExrSelectionLayer<'a>>::new();
    for channel in raw_channels {
        let Some((layer_name, pass_name, id)) =
            split_exr_channel_name(&channel.name, has_xyz_channels)
        else {
            continue;
        };
        let layer_index = if let Some(index) = layer_indices.get(&layer_name) {
            *index
        } else {
            let index = layers.len();
            layer_indices.insert(layer_name.clone(), index);
            layers.push(ExrSelectionLayer { passes: Vec::new() });
            index
        };
        let view_id = views
            .iter()
            .position(|known| known == &channel.view)
            .unwrap_or(0);
        let layer = &mut layers[layer_index];
        if let Some(pass) = layer
            .passes
            .iter_mut()
            .find(|pass| pass.name == pass_name && pass.view == channel.view)
        {
            pass.channels.push(ExrSelectionChannel {
                channel: channel.channel,
                id,
            });
        } else {
            layer.passes.push(ExrSelectionPass {
                name: pass_name,
                view: channel.view,
                view_id,
                width: channel.width,
                height: channel.height,
                channels: vec![ExrSelectionChannel {
                    channel: channel.channel,
                    id,
                }],
            });
        }
    }
    for layer in &mut layers {
        layer.passes.sort_by(compare_exr_passes);
    }
    layers
}

fn remove_exr_view_name(name: &str, views: &[String]) -> (String, String) {
    let Some((prefix, channel)) = name.rsplit_once('.') else {
        return (name.to_owned(), String::new());
    };
    for view in views {
        if prefix == view {
            return (channel.to_owned(), view.clone());
        }
        if let Some(layer_pass) = prefix
            .strip_suffix(view)
            .and_then(|rest| rest.strip_suffix('.'))
        {
            let mut name_without_view = String::with_capacity(layer_pass.len() + channel.len() + 1);
            if !layer_pass.is_empty() {
                name_without_view.push_str(layer_pass);
                name_without_view.push('.');
            }
            name_without_view.push_str(channel);
            return (name_without_view, view.clone());
        }
    }
    (name.to_owned(), String::new())
}

fn split_exr_channel_name(name: &str, has_xyz_channels: bool) -> Option<(String, String, char)> {
    let (prefix, channel_name) = name.rsplit_once('.').unwrap_or(("", name));
    if channel_name.is_empty() {
        return None;
    }
    let id = exr_channel_id(channel_name)?;
    if prefix.is_empty() {
        let pass_name =
            if id == 'Z' && (!has_xyz_channels || channel_name.eq_ignore_ascii_case("depth")) {
                "Depth"
            } else if id == 'Y' && !has_xyz_channels {
                channel_name
            } else if matches!(id, 'R' | 'G' | 'B' | 'A' | 'V' | 'X' | 'Y' | 'Z') {
                "Combined"
            } else {
                channel_name
            };
        return Some((String::new(), pass_name.to_owned(), id));
    }
    let (layer_name, pass_name) = prefix.rsplit_once('.').unwrap_or(("", prefix));
    if pass_name.is_empty() {
        return None;
    }
    Some((layer_name.to_owned(), pass_name.to_owned(), id))
}

fn exr_channel_id(name: &str) -> Option<char> {
    let first = name.chars().next()?.to_ascii_uppercase();
    if name.len() == 1 {
        return Some(first);
    }
    if name.len() == 2 {
        let second = name.chars().nth(1)?.to_ascii_uppercase();
        return Some(
            if matches!(second, 'X' | 'Y' | 'Z' | 'R' | 'G' | 'B' | 'U' | 'V' | 'A') {
                second
            } else {
                'X'
            },
        );
    }
    Some(if name.eq_ignore_ascii_case("red") {
        'R'
    } else if name.eq_ignore_ascii_case("green") {
        'G'
    } else if name.eq_ignore_ascii_case("blue") {
        'B'
    } else if name.eq_ignore_ascii_case("alpha") {
        'A'
    } else if name.eq_ignore_ascii_case("depth") {
        'Z'
    } else {
        'X'
    })
}

/// Blender's `RE_PASSNAME_*` tokens and their `SCE_PASS_*` bits, in `passtype_from_name` order
/// (`render_result.cc`, `DNA_scene_enums.h`). The first token that starts the pass name gives its type.
const BLENDER_PASS_TYPES: &[(&str, u32)] = &[
    ("Combined", 1 << 0),
    ("Depth", 1 << 1),
    ("Vector", 1 << 9),
    ("Normal", 1 << 8),
    ("UV", 1 << 12),
    ("Emission", 1 << 16),
    ("Shadow", 1 << 5),
    ("Ambient Occlusion", 1 << 6),
    ("Environment", 1 << 17),
    ("Object Index", 1 << 11),
    ("Material Index", 1 << 18),
    ("Mist", 1 << 14),
    ("Diffuse Direct", 1 << 19),
    ("Diffuse Indirect", 1 << 20),
    ("Diffuse Color", 1 << 21),
    ("Glossy Direct", 1 << 22),
    ("Glossy Indirect", 1 << 23),
    ("Glossy Color", 1 << 24),
    ("Transmission Direct", 1 << 25),
    ("Transmission Indirect", 1 << 26),
    ("Transmission Color", 1 << 27),
    ("Subsurface Direct", 1 << 28),
    ("Subsurface Indirect", 1 << 29),
    ("Subsurface Color", 1 << 30),
];

/// Blender's `passtype_from_name`: the type of the first token of `name` that its `RE_PASSNAME_*`
/// starts with, where an empty token matches the first entry (`STREQLEN` with length 0).
fn blender_pass_type(name: &str) -> u32 {
    let token = name.split('.').next().unwrap_or_default();
    BLENDER_PASS_TYPES
        .iter()
        .find(|(pass, _)| pass.as_bytes().starts_with(token.as_bytes()))
        .map_or(0, |(_, bit)| *bit)
}

/// Blender's `order_render_passes` (`render_result.cc`): whether `left` sorts after `right`.
/// Typed passes come before untyped ones, typed passes sort by type, untyped passes by name, and
/// equal passes put the left view first, then the right view, then by view id.
fn exr_pass_sorts_after(left: &ExrSelectionPass<'_>, right: &ExrSelectionPass<'_>) -> bool {
    let left_type = blender_pass_type(&left.name);
    let right_type = blender_pass_type(&right.name);
    if right_type != 0 && left_type == 0 {
        return true;
    }
    if left_type != 0 && right_type == 0 {
        return false;
    }
    if left_type != 0 && right_type != 0 {
        if left_type != right_type {
            return left_type > right_type;
        }
    } else if left.name != right.name {
        return left.name > right.name;
    }
    if left.view == "left" {
        return false;
    }
    if right.view == "left" {
        return true;
    }
    if left.view == "right" {
        return false;
    }
    if right.view == "right" {
        return true;
    }
    left.view_id < right.view_id
}

fn compare_exr_passes(
    left: &ExrSelectionPass<'_>,
    right: &ExrSelectionPass<'_>,
) -> std::cmp::Ordering {
    if exr_pass_sorts_after(left, right) {
        std::cmp::Ordering::Greater
    } else if exr_pass_sorts_after(right, left) {
        std::cmp::Ordering::Less
    } else {
        std::cmp::Ordering::Equal
    }
}

/// Blender's `BKE_image_multilayer_index` with `image_render_pass_get` (`imbuf` `image.cc`). The
/// layer is taken by index and the pass by its index in that layer. A view other than 0 takes the
/// next pass after the requested one that has the same name and view. A pass that is not found
/// falls back to the layer's first pass. A layer index beyond the layers has no pass.
fn select_exr_render_pass<'layers, 'image>(
    layers: &'layers [ExrSelectionLayer<'image>],
    selection: ImageUserSelection,
) -> Option<&'layers ExrSelectionPass<'image>> {
    let layer = layers.get(usize::from(selection.layer))?;
    let requested = usize::from(selection.pass);
    let view = usize::from(selection.view);
    let mut found = None;
    let mut requested_name: Option<&str> = None;
    for (index, candidate) in layer.passes.iter().enumerate() {
        if index == requested {
            found = Some(index);
            if view == 0 {
                break;
            }
            requested_name = Some(candidate.name.as_str());
        } else if let Some(name) = requested_name.filter(|name| !name.is_empty())
            && candidate.name == name
            && candidate.view_id == view
        {
            found = Some(index);
            break;
        }
    }
    found
        .map(|index| &layer.passes[index])
        .or_else(|| layer.passes.first())
}

fn exr_pass_channel_layout(pass: &ExrSelectionPass<'_>) -> ([Option<usize>; 4], [char; 4]) {
    let mut slots = [None; 4];
    let mut ids = ['\0'; 4];
    let count = pass.channels.len();
    if count == 1 {
        slots[0] = Some(0);
        ids[0] = pass.channels[0].id;
        return (slots, ids);
    }
    if count == 3 || count == 4 {
        let first_three = &pass.channels[..3];
        let lookup = if first_three.iter().any(|channel| channel.id == 'B') {
            [('R', 0), ('G', 1), ('B', 2), ('A', 3)]
        } else if first_three.iter().any(|channel| channel.id == 'Y') {
            [('X', 0), ('Y', 1), ('Z', 2), ('W', 3)]
        } else {
            [('U', 0), ('V', 1), ('A', 2), ('\0', 3)]
        };
        for (channel_index, channel) in pass.channels.iter().enumerate() {
            let slot = lookup
                .iter()
                .find_map(|(id, slot)| (channel.id == *id).then_some(*slot))
                .unwrap_or(0);
            slots[slot] = Some(channel_index);
            ids[slot] = channel.id;
        }
        return (slots, ids);
    }
    for (index, channel) in pass.channels.iter().enumerate().take(4) {
        slots[index] = Some(index);
        ids[index] = channel.id;
    }
    (slots, ids)
}

fn is_exr_color_pass(channel_count: usize, ids: &[char; 4]) -> bool {
    if channel_count > ids.len() {
        return false;
    }
    let channel_ids = ids.iter().take(channel_count).copied().collect::<String>();
    matches!(channel_ids.as_str(), "RGB" | "RGBA" | "R" | "G" | "B" | "A")
}
fn image_blob_bytes<'a>(
    blob: Option<&str>,
    image: &Image,
    root: &Path,
    staged: &'a BTreeMap<String, Vec<u8>>,
) -> Result<Cow<'a, [u8]>> {
    if let Some(blob) = blob {
        return blob_bytes(blob, root, staged);
    }
    let source_path = image.source_path.as_deref().ok_or_else(|| {
        PotError::new(
            ErrorCode::FileNotFound,
            "image has no packed blob or linked source path",
        )
    })?;
    let source_path_ref = Path::new(source_path);
    let resolved_path = if source_path_ref.is_absolute() {
        source_path_ref.to_path_buf()
    } else {
        root.join(source_path_ref)
    };
    let bytes = fs::read(&resolved_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PotError::with_details(
                ErrorCode::FileNotFound,
                "linked image source file is missing",
                serde_json::json!({"path":resolved_path}),
            )
        } else {
            PotError::io(&error)
        }
    })?;
    let expected = image.source_hash.as_deref().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "linked image has no registered source hash",
        )
    })?;
    let actual = hash::sha256(&bytes);
    if actual != expected {
        return Err(PotError::with_details(
            ErrorCode::AssetChanged,
            "linked image source has changed since it was registered",
            serde_json::json!({"feature_id":"asset.changed","path":resolved_path,"expected_hash":expected,"actual_hash":actual}),
        ));
    }
    Ok(Cow::Owned(bytes))
}

/// The encoded bytes an image is loaded from, read as the loader reads them: its packed blob, or its linked source file,
/// with the same hash checks. Scene upgrades read an image's format through this.
pub(crate) fn image_source_bytes(image: &Image, root: &Path) -> Result<Vec<u8>> {
    let staged = BTreeMap::new();
    image_blob_bytes(image.blob.as_deref(), image, root, &staged).map(Cow::into_owned)
}

fn blob_bytes<'a>(
    blob: &str,
    root: &Path,
    staged: &'a BTreeMap<String, Vec<u8>>,
) -> Result<Cow<'a, [u8]>> {
    let Some(hex) = blob.strip_prefix("sha256:") else {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "image blob hash is not sha256",
        ));
    };
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "image blob hash is malformed",
        ));
    }
    if let Some(bytes) = staged.get(blob) {
        if hash::sha256(bytes) != blob {
            return Err(PotError::with_details(
                ErrorCode::ValidationFailed,
                "staged image asset content does not match its hash",
                serde_json::json!({"hash":blob}),
            ));
        }
        return Ok(Cow::Borrowed(bytes));
    }
    let path = root.join("assets").join("sha256").join(hex).join("blob");
    let bytes = fs::read(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PotError::with_details(
                ErrorCode::FileNotFound,
                "image asset blob is missing",
                serde_json::json!({"path":path,"hash":blob}),
            )
        } else {
            PotError::io(&error)
        }
    })?;
    if hash::sha256(&bytes) != blob {
        return Err(PotError::with_details(
            ErrorCode::ValidationFailed,
            "image asset blob content does not match its hash",
            serde_json::json!({"path":path,"hash":blob}),
        ));
    }
    Ok(Cow::Owned(bytes))
}

fn checked_pixel_count(width: u32, height: u32) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "image dimensions must be positive",
        ));
    }
    let count = bounded_pixel_product(width, height).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image exceeds the maximum pixel count",
        )
    })?;
    usize::try_from(count).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image dimensions exceed platform limits",
        )
    })
}

fn bounded_pixel_product(width: u32, height: u32) -> Option<u64> {
    u64::from(width)
        .checked_mul(u64::from(height))
        .filter(|count| *count <= MAX_IMAGE_PIXELS)
}

#[cfg(kani)]
mod verification {
    use super::{MAX_IMAGE_PIXELS, bounded_pixel_product};

    #[kani::proof]
    fn bounded_pixel_product_matches_the_limited_dimension_product() {
        let width: u32 = kani::any();
        let height: u32 = kani::any();
        let product = u64::from(width) * u64::from(height);
        let result = bounded_pixel_product(width, height);

        if product <= MAX_IMAGE_PIXELS {
            assert_eq!(result, Some(product));
        } else {
            assert!(result.is_none());
        }
    }
}

fn validate_pixels(width: u32, height: u32, pixels: &[[f64; 4]]) -> Result<()> {
    let expected = checked_pixel_count(width, height)?;
    if expected != pixels.len() {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "image dimensions and pixel count are inconsistent",
        ));
    }
    if pixels
        .iter()
        .flatten()
        .any(|component| !component.is_finite())
    {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "image pixels must be finite",
        ));
    }
    Ok(())
}

/// Validates and reads dimensions from a PNG without converting decoded texels to floats.
pub fn decode_png_dimensions(bytes: &[u8]) -> Result<(u32, u32)> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            format!("PNG image is invalid: {error}"),
            serde_json::json!({"feature_id":"image.encoding.png"}),
        )
    })?;
    let width = reader.info().width;
    let height = reader.info().height;
    checked_pixel_count(width, height)?;
    let buffer_size = reader.output_buffer_size().ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "PNG output buffer exceeds platform limits",
        )
    })?;
    let mut buffer = vec![0_u8; buffer_size];
    let frame = reader.next_frame(&mut buffer).map_err(|error| {
        PotError::new(
            ErrorCode::InvalidArgument,
            format!("PNG image decode failed: {error}"),
        )
    })?;
    if frame.width != width || frame.height != height {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "PNG frame dimensions changed during decoding",
        ));
    }
    Ok((width, height))
}

fn decode_png(bytes: &[u8], colorspace: ImageColorspace) -> Result<(u32, u32, Vec<[f64; 4]>)> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            format!("PNG image is invalid: {error}"),
            serde_json::json!({"feature_id":"image.encoding.png.invalid_data"}),
        )
    })?;
    let width = reader.info().width;
    let height = reader.info().height;
    let pixel_count = checked_pixel_count(width, height)?;
    let buffer_size = reader.output_buffer_size().ok_or_else(|| {
        encoding_error(
            ErrorCode::LimitExceeded,
            "png",
            "invalid_data",
            "PNG output buffer exceeds platform limits",
        )
    })?;
    let mut buffer = vec![0_u8; buffer_size];
    let frame = reader.next_frame(&mut buffer).map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            format!("PNG image decode failed: {error}"),
            serde_json::json!({"feature_id":"image.encoding.png.invalid_data"}),
        )
    })?;
    if frame.width != width || frame.height != height {
        return Err(encoding_error(
            ErrorCode::InvalidArgument,
            "png",
            "invalid_data",
            "PNG frame dimensions changed during decoding",
        ));
    }
    let channels = match frame.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => {
            return Err(encoding_error(
                ErrorCode::InvalidArgument,
                "png",
                "invalid_data",
                "PNG palette was not expanded",
            ));
        }
    };
    let bytes_per_sample = match frame.bit_depth {
        png::BitDepth::Eight => 1,
        png::BitDepth::Sixteen => 2,
        _ => {
            return Err(encoding_error(
                ErrorCode::UnsupportedFeature,
                "png",
                "unsupported_bit_depth",
                "PNG bit depth is unsupported",
            ));
        }
    };
    let sample_size = channels * bytes_per_sample;
    let decoded = &buffer[..frame.buffer_size()];
    if decoded.len() != pixel_count.saturating_mul(sample_size) {
        return Err(encoding_error(
            ErrorCode::InvalidArgument,
            "png",
            "invalid_data",
            "PNG decoded pixel buffer has an invalid size",
        ));
    }
    let mut pixels = Vec::with_capacity(pixel_count);
    for sample in decoded.chunks_exact(sample_size) {
        let value = |channel: usize| {
            let start = channel * bytes_per_sample;
            if bytes_per_sample == 1 {
                f64::from(sample[start]) / 255.0
            } else {
                f64::from(u16::from_be_bytes([sample[start], sample[start + 1]])) / 65_535.0
            }
        };
        let (red, green, blue, alpha) = match frame.color_type {
            png::ColorType::Grayscale => {
                let gray = value(0);
                (gray, gray, gray, 1.0)
            }
            png::ColorType::GrayscaleAlpha => {
                let gray = value(0);
                (gray, gray, gray, value(1))
            }
            png::ColorType::Rgb => (value(0), value(1), value(2), 1.0),
            png::ColorType::Rgba => (value(0), value(1), value(2), value(3)),
            png::ColorType::Indexed => {
                return Err(encoding_error(
                    ErrorCode::InvalidArgument,
                    "png",
                    "invalid_data",
                    "PNG palette was not expanded",
                ));
            }
        };
        pixels.push([red, green, blue, alpha]);
    }
    let pixels = apply_colorspace(pixels, colorspace);
    validate_pixels(width, height, &pixels)?;
    Ok((width, height, pixels))
}
