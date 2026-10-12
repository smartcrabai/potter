//! Reference photos sent to the modeler and reviewer agents.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, bail};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use image::imageops::FilterType;
use image::{DynamicImage, ImageReader};

/// Longest edge a reference is sent at. Keeps every reference within provider image limits and
/// bounds the tokens each review pays for.
const MAX_EDGE_PX: u32 = 1568;

/// Lossy WebP quality (0-100). High enough that shading and edges survive, low enough that photos
/// shrink well below their decoded size.
const WEBP_QUALITY: f32 = 85.0;

/// Decodes each reference, downscales it to fit `MAX_EDGE_PX`, and encodes it as lossy WebP.
///
/// Returns `("image/webp", base64)` pairs in input order. Every reference is re-encoded, including
/// small ones, so providers always receive `image/webp`. Unsupported extensions are rejected before
/// any image is decoded.
pub(crate) fn load_references(paths: &[PathBuf]) -> anyhow::Result<Vec<(String, String)>> {
    for path in paths {
        check_extension(path)?;
    }
    paths
        .iter()
        .map(|path| {
            let decoded = decode_reference(path)?;
            let fitted = fit_within_max_edge(decoded);
            let webp = encode_webp(&fitted)
                .with_context(|| format!("encoding reference image {} as WebP", path.display()))?;
            Ok(("image/webp".to_owned(), STANDARD.encode(&*webp)))
        })
        .collect()
}

fn check_extension(path: &Path) -> anyhow::Result<()> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some("png" | "jpg" | "jpeg" | "webp" | "gif") => Ok(()),
        _ => bail!(
            "unsupported reference image extension for {} (expected png, jpg, jpeg, webp, or gif)",
            path.display()
        ),
    }
}

fn decode_reference(path: &Path) -> anyhow::Result<DynamicImage> {
    ImageReader::open(path)
        .with_context(|| format!("reading reference image {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("reading reference image {}", path.display()))?
        .decode()
        .with_context(|| format!("decoding reference image {}", path.display()))
}

/// Shrinks the image so its longer edge is at most `MAX_EDGE_PX`, keeping the aspect ratio.
/// Images already within the limit are returned unchanged, so they are never upscaled.
fn fit_within_max_edge(decoded: DynamicImage) -> DynamicImage {
    let (width, height) = (decoded.width(), decoded.height());
    if width.max(height) <= MAX_EDGE_PX {
        return decoded;
    }
    let (new_width, new_height) = if width >= height {
        (MAX_EDGE_PX, scale_edge(height, width))
    } else {
        (scale_edge(width, height), MAX_EDGE_PX)
    };
    decoded.resize_exact(new_width, new_height, FilterType::Lanczos3)
}

/// Returns `short * MAX_EDGE_PX / long`, rounded to nearest and at least 1 px. Requires
/// `short <= long` and `long > MAX_EDGE_PX`, so the result never exceeds `MAX_EDGE_PX`.
fn scale_edge(short: u32, long: u32) -> u32 {
    let numerator = u64::from(short) * u64::from(MAX_EDGE_PX) + u64::from(long) / 2;
    (numerator / u64::from(long)).max(1) as u32
}

fn encode_webp(image: &DynamicImage) -> anyhow::Result<webp::WebPMemory> {
    let encoded = if image.color().has_alpha() {
        let rgba = image.to_rgba8();
        webp::Encoder::from_rgba(rgba.as_raw(), rgba.width(), rgba.height())
            .encode_simple(false, WEBP_QUALITY)
    } else {
        let rgb = image.to_rgb8();
        webp::Encoder::from_rgb(rgb.as_raw(), rgb.width(), rgb.height())
            .encode_simple(false, WEBP_QUALITY)
    };
    encoded.map_err(|error| anyhow::anyhow!("WebP encoder failed: {error:?}"))
}

#[cfg(test)]
mod tests {
    use anyhow::Context as _;
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;
    use image::{ImageFormat, Rgb, RgbImage};

    use super::load_references;

    fn decode_webp_reference(data: &str) -> anyhow::Result<image::DynamicImage> {
        let bytes = STANDARD.decode(data).context("decoding base64 reference")?;
        Ok(image::load_from_memory_with_format(
            &bytes,
            ImageFormat::WebP,
        )?)
    }

    #[test]
    fn large_reference_is_downscaled_to_max_edge_as_webp() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("wide.png");
        RgbImage::from_pixel(4000, 1000, Rgb([200, 120, 40])).save(&path)?;

        let references = load_references(std::slice::from_ref(&path))?;
        let [(media_type, data)] = references.as_slice() else {
            anyhow::bail!("expected exactly one reference, got {}", references.len());
        };

        assert_eq!(media_type, "image/webp");
        let decoded = decode_webp_reference(data)?;
        assert_eq!((decoded.width(), decoded.height()), (1568, 392));
        Ok(())
    }

    #[test]
    fn small_reference_is_not_upscaled() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("small.png");
        RgbImage::from_pixel(100, 50, Rgb([10, 20, 30])).save(&path)?;

        let references = load_references(std::slice::from_ref(&path))?;
        let [(media_type, data)] = references.as_slice() else {
            anyhow::bail!("expected exactly one reference, got {}", references.len());
        };

        assert_eq!(media_type, "image/webp");
        let decoded = decode_webp_reference(data)?;
        assert_eq!((decoded.width(), decoded.height()), (100, 50));
        Ok(())
    }
}
