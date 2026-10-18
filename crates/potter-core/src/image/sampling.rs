use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageInterpolation {
    #[default]
    Linear,
    Closest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageExtension {
    Repeat,
    Extend,
    Clip,
    Mirror,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageTileData {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f64; 4]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f64; 4]>,
    pub tiles: BTreeMap<u32, ImageTileData>,
    pub interpolation: ImageInterpolation,
}

const TRANSPARENT_BLACK: [f64; 4] = [0.0; 4];

/// Samples linear RGBA image pixels at a normalized UV coordinate.
///
/// Tile zero samples the base image directly. Positive tiles are treated as UDIM
/// base tiles, with integer UV offsets selecting neighboring tiles.
#[must_use]
pub fn sample(data: &ImageData, uv: [f64; 2], tile: u32) -> [f64; 4] {
    sample_with_interpolation(data, uv, tile, data.interpolation)
}

/// Samples an image with an explicit interpolation mode without copying the image data.
#[must_use]
pub fn sample_with_interpolation(
    data: &ImageData,
    uv: [f64; 2],
    tile: u32,
    interpolation: ImageInterpolation,
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

    match interpolation {
        ImageInterpolation::Linear => sample_linear(image.pixels, width, height, u, v),
        ImageInterpolation::Closest => sample_closest(image.pixels, width, height, u, v),
    }
}

/// Samples an image using separate UVs for UDIM routing and pixel lookup.
///
/// `routing_uv` selects the UDIM tile. `pixel_uv` is the full-image UV for tile zero and the
/// tile-local V-down UV for positive base tiles.
#[must_use]
pub fn sample_with_extension(
    data: &ImageData,
    routing_uv: [f64; 2],
    pixel_uv: [f64; 2],
    tile: u32,
    interpolation: ImageInterpolation,
    extension: ImageExtension,
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

    match interpolation {
        ImageInterpolation::Linear => sample_linear_with_extension(
            image.pixels,
            width,
            height,
            local_uv[0],
            local_uv[1],
            extension,
        ),
        ImageInterpolation::Closest => sample_closest_with_extension(
            image.pixels,
            width,
            height,
            local_uv[0],
            local_uv[1],
            extension,
        ),
    }
}

fn image_for_uv(data: &ImageData, uv: [f64; 2], tile: u32) -> Option<ImageTileDataRef<'_>> {
    if tile == 0 {
        Some(ImageTileDataRef {
            width: data.width,
            height: data.height,
            pixels: &data.pixels,
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
                    }
                } else {
                    ImageTileDataRef {
                        width: 0,
                        height: 0,
                        pixels: &[],
                    }
                }
            },
            |image| ImageTileDataRef {
                width: image.width,
                height: image.height,
                pixels: &image.pixels,
            },
        );
        Some(image)
    }
}

struct ImageTileDataRef<'a> {
    width: u32,
    height: u32,
    pixels: &'a [[f64; 4]],
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
fn sample_closest(pixels: &[[f64; 4]], width: usize, height: usize, u: f64, v: f64) -> [f64; 4] {
    let x = ((u * width as f64).floor() as usize).min(width - 1);
    let y = ((v * height as f64).floor() as usize).min(height - 1);
    pixel(pixels, width, x, y)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "validated image dimensions and clamped UVs keep conversions within exact texel bounds"
)]
fn sample_linear(pixels: &[[f64; 4]], width: usize, height: usize, u: f64, v: f64) -> [f64; 4] {
    let x = (u * width as f64 - 0.5).clamp(0.0, (width - 1) as f64);
    let y = (v * height as f64 - 0.5).clamp(0.0, (height - 1) as f64);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;

    let top = mix(
        pixel(pixels, width, x0, y0),
        pixel(pixels, width, x1, y0),
        tx,
    );
    let bottom = mix(
        pixel(pixels, width, x0, y1),
        pixel(pixels, width, x1, y1),
        tx,
    );
    mix(top, bottom, ty)
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
    pixel_with_extension(pixels, width, height, x, y, extension)
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

    let top = mix(
        pixel_with_extension(pixels, width, height, x0, y0, extension),
        pixel_with_extension(pixels, width, height, x1, y0, extension),
        tx,
    );
    let bottom = mix(
        pixel_with_extension(pixels, width, height, x0, y1, extension),
        pixel_with_extension(pixels, width, height, x1, y1, extension),
        tx,
    );
    mix(top, bottom, ty)
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
        ImageData, ImageExtension, ImageInterpolation, ImageTileData, sample,
        sample_with_extension, sample_with_interpolation,
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
        }
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
