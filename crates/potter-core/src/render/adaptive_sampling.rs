use glam::Vec3;

use crate::error::{ErrorCode, PotError, Result};

const ADAPTIVE_STEP: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct AdaptiveSampling {
    pub threshold: f32,
    pub min_samples: u32,
}

impl AdaptiveSampling {
    pub(super) fn new(max_samples: u32, mut threshold: f32, mut min_samples: u32) -> Self {
        if threshold == 0.0 && max_samples > 0 {
            threshold = (1.0 / max_samples as f32).max(0.001);
        }

        if min_samples == 0 && threshold > 0.0 {
            min_samples = (16.0 / threshold.powf(0.3)).ceil() as u32;
        }

        Self {
            threshold: threshold * 5.0,
            min_samples: min_samples.max(4),
        }
    }

    pub(super) fn align_batch_end(self, completed: u32, proposed_end: u32) -> u32 {
        if proposed_end <= completed {
            return completed;
        }

        let first_filter_sample = self.min_samples.saturating_add(1) | (ADAPTIVE_STEP - 1);
        if proposed_end <= first_filter_sample {
            return proposed_end;
        }

        let next_filter_sample = first_filter_sample.max(completed | (ADAPTIVE_STEP - 1));
        next_filter_sample.saturating_add(1).min(proposed_end)
    }

    pub(super) fn needs_filter(self, completed: u32) -> bool {
        completed.checked_sub(1).is_some_and(|sample| {
            sample > self.min_samples && sample & (ADAPTIVE_STEP - 1) == ADAPTIVE_STEP - 1
        })
    }

    pub(super) fn filter_pixels(
        self,
        pixels: &mut [AdaptivePixel],
        width: u32,
        height: u32,
        exposure: f32,
        count: impl Fn(usize) -> u32 + Sync,
    ) -> Result<usize> {
        let width = usize::try_from(width)
            .map_err(|_| allocation_error("adaptive sampling width exceeds addressable memory"))?;
        let height = usize::try_from(height)
            .map_err(|_| allocation_error("adaptive sampling height exceeds addressable memory"))?;
        let pixel_count = width.checked_mul(height).ok_or_else(|| {
            allocation_error("adaptive sampling dimensions exceed addressable memory")
        })?;
        if pixel_count != pixels.len() {
            return Err(PotError::invalid_argument(
                "adaptive sampling pixel buffer does not match its dimensions",
            ));
        }
        if width == 0 || height == 0 {
            return Ok(0);
        }

        let available_workers =
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        let raw_active = filter_rows_parallel(
            pixels,
            width,
            height,
            available_workers.min(height),
            exposure,
            self.threshold,
            &count,
        )?;
        if raw_active == 0 {
            return Ok(0);
        }

        let column_bands = split_column_bands(pixels, width, height, available_workers.min(width))?;
        filter_y_parallel(column_bands)?;
        Ok(raw_active)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct AdaptivePixel {
    combined: Vec3,
    class_a_x2: Vec3,
    converged: bool,
}

impl AdaptivePixel {
    pub(super) fn add_contribution(&mut self, sample_index: u32, contribution: Vec3) {
        self.combined += contribution;
        if sample_is_class_a(sample_index) {
            self.class_a_x2 += contribution * 2.0;
        }
    }

    pub(super) fn needs_sample(&self) -> bool {
        !self.converged
    }

    fn convergence_check(&mut self, count: u32, exposure: f32, threshold: f32) -> bool {
        if self.converged {
            return true;
        }

        let sample = count as f32;
        let intensity_scale = exposure / sample;
        let error_difference = ((self.combined.x - self.class_a_x2.x).abs()
            + (self.combined.y - self.class_a_x2.y).abs()
            + (self.combined.z - self.class_a_x2.z).abs())
            * intensity_scale;
        let intensity = ((self.combined.x + self.combined.y) + self.combined.z) * intensity_scale;
        let error_normalize = if intensity < 1.0 {
            intensity.sqrt()
        } else {
            intensity
        };
        let error = error_difference / (0.0001 + error_normalize);
        self.converged = error < threshold;
        self.converged
    }
}

fn sample_is_class_a(sample_index: u32) -> bool {
    (sample_index & 0xaaaa_aaaa).count_ones() & 1 != 0
}

fn filter_rows_parallel<F>(
    pixels: &mut [AdaptivePixel],
    width: usize,
    height: usize,
    worker_count: usize,
    exposure: f32,
    threshold: f32,
    count: &F,
) -> Result<usize>
where
    F: Fn(usize) -> u32 + Sync,
{
    if worker_count == 1 {
        return Ok(filter_rows(pixels, width, 0, exposure, threshold, count));
    }

    let rows_per_band = height.div_ceil(worker_count);
    let pixels_per_band = rows_per_band * width;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(worker_count)
            .map_err(|_| allocation_error("adaptive sampling row worker allocation failed"))?;
        for (band_index, rows) in pixels.chunks_mut(pixels_per_band).enumerate() {
            let start_y = band_index * rows_per_band;
            handles.push(std::thread::Builder::new().spawn_scoped(scope, move || {
                filter_rows(rows, width, start_y, exposure, threshold, count)
            }));
        }
        join_filter_workers(handles)
    })
}

fn filter_rows<F>(
    pixels: &mut [AdaptivePixel],
    width: usize,
    start_y: usize,
    exposure: f32,
    threshold: f32,
    count: &F,
) -> usize
where
    F: Fn(usize) -> u32 + Sync,
{
    let mut raw_active = 0;
    for (local_y, row) in pixels.chunks_exact_mut(width).enumerate() {
        let mut row_active = 0;
        let row_start = (start_y + local_y) * width;
        for (x, pixel) in row.iter_mut().enumerate() {
            if !pixel.converged
                && !pixel.convergence_check(count(row_start + x), exposure, threshold)
            {
                row_active += 1;
            }
        }
        raw_active += row_active;
        if row_active != 0 {
            filter_x_row(row);
        }
    }
    raw_active
}

fn filter_x_row(row: &mut [AdaptivePixel]) {
    let mut previous: Option<&mut AdaptivePixel> = None;
    let mut prev = false;
    for pixel in row.iter_mut() {
        if pixel.converged {
            if prev {
                pixel.converged = false;
            }
            prev = false;
        } else {
            if !prev && let Some(previous_pixel) = previous.as_deref_mut() {
                previous_pixel.converged = false;
            }
            prev = true;
        }
        previous = Some(pixel);
    }
}

fn split_column_bands(
    pixels: &mut [AdaptivePixel],
    width: usize,
    height: usize,
    band_count: usize,
) -> Result<Vec<Vec<&mut [AdaptivePixel]>>> {
    let mut bands = Vec::new();
    bands
        .try_reserve_exact(band_count)
        .map_err(|_| allocation_error("adaptive sampling column-band allocation failed"))?;
    for _ in 0..band_count {
        let mut rows = Vec::new();
        rows.try_reserve_exact(height)
            .map_err(|_| allocation_error("adaptive sampling column-band allocation failed"))?;
        bands.push(rows);
    }

    for row in pixels.chunks_mut(width) {
        let mut remaining = row;
        for (index, band) in bands.iter_mut().enumerate() {
            let count = band_width(width, band_count, index);
            let (columns, rest) = remaining.split_at_mut(count);
            band.push(columns);
            remaining = rest;
        }
    }
    Ok(bands)
}

fn band_width(total: usize, band_count: usize, index: usize) -> usize {
    let base = total / band_count;
    let remainder = total % band_count;
    base + usize::from(index < remainder)
}

fn filter_y_parallel(bands: Vec<Vec<&mut [AdaptivePixel]>>) -> Result<()> {
    let band_count = bands.len();
    if band_count == 1 {
        if let Some(rows) = bands.into_iter().next() {
            filter_y_columns(rows);
        }
        return Ok(());
    }

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(band_count)
            .map_err(|_| allocation_error("adaptive sampling column worker allocation failed"))?;
        for rows in bands {
            handles.push(std::thread::Builder::new().spawn_scoped(scope, move || {
                filter_y_columns(rows);
                0
            }));
        }
        join_filter_workers(handles).map(|_| ())
    })
}

fn filter_y_columns(mut rows: Vec<&mut [AdaptivePixel]>) {
    let Some((first_row, remaining_rows)) = rows.split_first_mut() else {
        return;
    };
    for (x, first_pixel) in first_row.iter_mut().enumerate() {
        let mut prev = !first_pixel.converged;
        let mut previous_pixel = Some(first_pixel);
        for row in remaining_rows.iter_mut() {
            let pixel = &mut row[x];
            if pixel.converged {
                if prev {
                    pixel.converged = false;
                }
                prev = false;
            } else {
                if !prev && let Some(previous_pixel) = previous_pixel.as_deref_mut() {
                    previous_pixel.converged = false;
                }
                prev = true;
            }
            previous_pixel = Some(pixel);
        }
    }
}

fn join_filter_workers(
    handles: Vec<std::io::Result<std::thread::ScopedJoinHandle<'_, usize>>>,
) -> Result<usize> {
    let mut raw_active = 0;
    let mut first_error = None;
    for handle in handles {
        match handle {
            Ok(handle) => match handle.join() {
                Ok(count) => raw_active += count,
                Err(_) if first_error.is_none() => {
                    first_error = Some(PotError::new(
                        ErrorCode::InternalError,
                        "adaptive sampling filter worker panicked",
                    ));
                }
                Err(_) => {}
            },
            Err(error) if first_error.is_none() => {
                first_error = Some(PotError::new(
                    ErrorCode::InternalError,
                    format!("cannot start an adaptive sampling filter worker: {error}"),
                ));
            }
            Err(_) => {}
        }
    }
    first_error.map_or(Ok(raw_active), Err)
}

fn allocation_error(message: &str) -> PotError {
    PotError::new(ErrorCode::LimitExceeded, message)
}

#[cfg(test)]
mod tests {
    use super::{AdaptivePixel, AdaptiveSampling};
    use glam::Vec3;

    #[test]
    fn class_a_uses_actual_resumed_sample_indices() {
        let mut pixel = AdaptivePixel::default();
        for sample_index in 79..84 {
            pixel.add_contribution(sample_index, Vec3::ONE);
        }

        assert_eq!(
            [79, 80, 81, 82, 83].map(super::sample_is_class_a),
            [false, false, false, true, true],
        );
        assert_eq!(pixel.combined, Vec3::splat(5.0));
        assert_eq!(pixel.class_a_x2, Vec3::splat(4.0));
        assert!(pixel.needs_sample());
    }

    #[test]
    fn adaptive_filter_schedule_handles_automatic_minimum_and_overflow() {
        let adaptive = AdaptiveSampling::new(100, 0.0, 0);
        assert!(!adaptive.needs_filter(0));
        assert!(!adaptive.needs_filter(16));
        assert!(adaptive.needs_filter(80));
        assert!(adaptive.needs_filter(96));
        assert_eq!(adaptive.align_batch_end(64, 200), 80);
        assert_eq!(adaptive.align_batch_end(80, 200), 96);

        assert_eq!(
            adaptive.align_batch_end(u32::MAX - 16, u32::MAX),
            u32::MAX - 15
        );
        assert_eq!(adaptive.align_batch_end(u32::MAX - 15, u32::MAX), u32::MAX);
        assert!(!adaptive.needs_filter(u32::MAX));
        assert!(adaptive.needs_filter(u32::MAX - 15));
    }

    #[test]
    fn convergence_uses_f32_brightness_exposure_and_source_comparisons() {
        let adaptive = AdaptiveSampling::new(1, 0.0036, 4);
        let make_pixel = || AdaptivePixel {
            combined: Vec3::new(0.5, 0.0, 0.0),
            class_a_x2: Vec3::new(0.49, 0.0, 0.0),
            converged: false,
        };

        let mut dim = [make_pixel()];
        assert_eq!(adaptive.filter_pixels(&mut dim, 1, 1, 1.0, |_| 1), Ok(0));
        assert!(dim[0].converged);

        let mut exposure_crosses_one = [make_pixel()];
        assert_eq!(
            adaptive.filter_pixels(&mut exposure_crosses_one, 1, 1, 2.0, |_| 1),
            Ok(1)
        );
        assert!(!exposure_crosses_one[0].converged);

        let mut zero = [AdaptivePixel {
            combined: Vec3::ZERO,
            class_a_x2: Vec3::ZERO,
            converged: false,
        }];
        assert_eq!(adaptive.filter_pixels(&mut zero, 1, 1, 1.0, |_| 1), Ok(0));
        assert!(zero[0].converged);

        let mut negative = [AdaptivePixel {
            combined: Vec3::new(-1.0, 0.0, 0.0),
            class_a_x2: Vec3::ZERO,
            converged: false,
        }];
        assert_eq!(
            adaptive.filter_pixels(&mut negative, 1, 1, 1.0, |_| 1),
            Ok(1)
        );
        assert!(!negative[0].converged);

        let mut nan = [AdaptivePixel {
            combined: Vec3::new(f32::NAN, 0.0, 0.0),
            class_a_x2: Vec3::ZERO,
            converged: false,
        }];
        assert_eq!(adaptive.filter_pixels(&mut nan, 1, 1, 1.0, |_| 1), Ok(1));
        assert!(!nan[0].converged);
    }

    #[test]
    fn previously_converged_pixels_early_out_without_reading_count() {
        let adaptive = AdaptiveSampling::new(1, 0.01, 4);
        let mut pixels = [AdaptivePixel {
            combined: Vec3::splat(f32::NAN),
            class_a_x2: Vec3::ZERO,
            converged: true,
        }];
        assert_eq!(
            adaptive.filter_pixels(&mut pixels, 1, 1, 1.0, |_| panic!("count must not be read")),
            Ok(0)
        );
        assert!(pixels[0].converged);
    }

    #[test]
    fn active_mask_dilates_exactly_three_by_three_and_clips_at_edges() {
        let adaptive = AdaptiveSampling::new(1, -1.0, 4);
        let mut pixels = vec![
            AdaptivePixel {
                converged: true,
                ..AdaptivePixel::default()
            };
            5 * 4
        ];
        pixels[5 + 2].converged = false;
        assert_eq!(
            adaptive.filter_pixels(&mut pixels, 5, 4, 1.0, |index| {
                assert_eq!(index, 5 + 2);
                1
            }),
            Ok(1)
        );
        for y in 0..4 {
            for x in 0..5 {
                assert_eq!(
                    pixels[y * 5 + x].needs_sample(),
                    (1..=3).contains(&x) && (0..=2).contains(&y),
                );
            }
        }

        let mut clipped = vec![
            AdaptivePixel {
                converged: true,
                ..AdaptivePixel::default()
            };
            3 * 3
        ];
        clipped[0].converged = false;
        assert_eq!(
            adaptive.filter_pixels(&mut clipped, 3, 3, 1.0, |_| 1),
            Ok(1)
        );
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(clipped[y * 3 + x].needs_sample(), x <= 1 && y <= 1,);
            }
        }
    }
}
