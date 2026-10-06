use crate::error::{ErrorCode, PotError, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 4]>,
}

impl Image {
    pub fn new(width: u32, height: u32, pixels: Vec<[f32; 4]>) -> Result<Self> {
        let expected = pixel_count(width, height)?;
        if pixels.len() != expected {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "image pixel count does not match its dimensions",
            ));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn filled(width: u32, height: u32, color: [f32; 4]) -> Result<Self> {
        let count = pixel_count(width, height)?;
        Ok(Self {
            width,
            height,
            pixels: vec![color; count],
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    Mix,
    Add,
    Multiply,
    Screen,
    Overlay,
}

#[must_use]
pub fn mix_rgb(mode: BlendMode, first: [f32; 4], second: [f32; 4]) -> [f32; 4] {
    std::array::from_fn(|channel| {
        let a = first[channel];
        let b = second[channel];
        match mode {
            BlendMode::Mix => b,
            BlendMode::Add => a + b,
            BlendMode::Multiply => a * b,
            BlendMode::Screen => 1.0 - (1.0 - a) * (1.0 - b),
            BlendMode::Overlay => {
                if a <= 0.5 {
                    2.0 * a * b
                } else {
                    1.0 - 2.0 * (1.0 - a) * (1.0 - b)
                }
            }
        }
    })
}

/// Separable Gaussian blur with periodic edge extension, which preserves image-wide means.
pub fn gaussian_blur(image: &Image, radius: u32) -> Result<Image> {
    if radius == 0 {
        return Ok(image.clone());
    }
    let size = usize::try_from(radius)
        .ok()
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "blur radius is too large"))?;
    let sigma = (f64::from(radius) / 2.0).max(0.5);
    let mut kernel = Vec::with_capacity(size);
    let mut total = 0.0;
    for offset in -(i64::from(radius))..=i64::from(radius) {
        let weight = (-0.5 * (offset as f64 / sigma).powi(2)).exp();
        kernel.push(weight);
        total += weight;
    }
    for weight in &mut kernel {
        *weight /= total;
    }

    let width = usize::try_from(image.width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image width exceeds platform limits",
        )
    })?;
    let height = usize::try_from(image.height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image height exceeds platform limits",
        )
    })?;
    let count = pixel_count(image.width, image.height)?;
    let mut horizontal = vec![[0.0_f32; 4]; count];
    let mut output = vec![[0.0_f32; 4]; count];

    for y in 0..height {
        for x in 0..width {
            let pixel_index = y * width + x;
            for (kernel_index, weight) in kernel.iter().enumerate() {
                let offset = i64::try_from(kernel_index).unwrap_or(i64::MAX) - i64::from(radius);
                let sample_x = (i64::try_from(x).unwrap_or(i64::MAX) + offset)
                    .rem_euclid(i64::try_from(width).unwrap_or(i64::MAX))
                    as usize;
                let sample = image.pixels[y * width + sample_x];
                for channel in 0..4 {
                    horizontal[pixel_index][channel] += sample[channel] * *weight as f32;
                }
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            let pixel_index = y * width + x;
            for (kernel_index, weight) in kernel.iter().enumerate() {
                let offset = i64::try_from(kernel_index).unwrap_or(i64::MAX) - i64::from(radius);
                let sample_y = (i64::try_from(y).unwrap_or(i64::MAX) + offset)
                    .rem_euclid(i64::try_from(height).unwrap_or(i64::MAX))
                    as usize;
                let sample = horizontal[sample_y * width + x];
                for channel in 0..4 {
                    output[pixel_index][channel] += sample[channel] * *weight as f32;
                }
            }
        }
    }
    Ok(Image {
        width: image.width,
        height: image.height,
        pixels: output,
    })
}

pub(super) fn pixel_count(width: u32, height: u32) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(PotError::invalid_argument(
            "image dimensions must be non-zero",
        ));
    }
    usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "image dimensions exceed platform limits",
            )
        })
}
