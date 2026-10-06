use std::{borrow::Cow, collections::BTreeMap, fs, io::Cursor, path::Path};

use serde::{Deserialize, Serialize};

use crate::{
    error::{ErrorCode, PotError, Result},
    hash,
    image::{ImageData, ImageInterpolation, ImageTileData},
    model::{Image, ImageColorspace},
};

const PIXEL_FORMAT: &str = "potter.rgba_f64.v1";
pub const MAX_IMAGE_PIXELS: u64 = 16_777_216;

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
    let base_bytes = image_blob_bytes(image.blob.as_deref(), image, root, staged)?;
    let (width, height, pixels) = decode_pixels(&base_bytes, image.colorspace)?;
    if width != image.width || height != image.height {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "image blob dimensions do not match its registry entry",
            serde_json::json!({"image":image.name,"expected":[image.width,image.height],"actual":[width,height]}),
        ));
    }
    let mut tiles = BTreeMap::new();
    for tile in &image.tiles {
        let bytes = blob_bytes(&tile.blob, root, staged)?;
        let (tile_width, tile_height, tile_pixels) = decode_pixels(&bytes, image.colorspace)?;
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
            },
        );
    }
    Ok(ImageData {
        width,
        height,
        pixels,
        tiles,
        interpolation,
    })
}

/// Decodes supported image encodings into linear RGBA pixels.
///
/// Eight-bit rasters default to sRGB; EXR, Radiance HDR, and floating-point TIFF default
/// to linear. EXR layers named `Combined` or `RGBA` are preferred over the first RGBA
/// layer. EXR RGB values are unpremultiplied when alpha is nonzero so the internal
/// representation uses the straight-alpha convention shared with PNG.
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
                *component = srgb_to_linear(*component);
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

fn decode_exr(bytes: &[u8]) -> Result<(u32, u32, Vec<[f64; 4]>)> {
    use exr::image::{AnyChannels, FlatSamples, Layer};
    use exr::prelude::ReadChannels;
    use exr::prelude::ReadLayers;
    type ExrLayer = Layer<AnyChannels<FlatSamples>>;
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

    let image = exr::prelude::read()
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
        })?;
    let has_rgb = |layer: &&ExrLayer| {
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
        let mut pixel = [
            exr_sample(red, index)?,
            exr_sample(green, index)?,
            exr_sample(blue, index)?,
            alpha.map_or(Ok(1.0), |channel| exr_sample(channel, index))?,
        ];
        let alpha_value = pixel[3];
        if alpha_value == 0.0 {
            pixel[..3].fill(0.0);
        } else {
            for component in &mut pixel[..3] {
                *component /= alpha_value;
            }
        }
        pixels.push(pixel);
    }
    validate_pixels(width, height, &pixels)?;
    Ok((width, height, pixels))
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
    let mut buffer = vec![0_u8; reader.output_buffer_size()];
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
    let mut buffer = vec![0_u8; reader.output_buffer_size()];
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

fn srgb_to_linear(value: f64) -> f64 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}
