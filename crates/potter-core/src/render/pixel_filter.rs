//! Cycles' film pixel filter, importance-sampled as Cycles does.
//!
//! Cycles builds an inverse-CDF table of the filter (`intern/cycles/scene/film.cpp` `filter_table`,
//! `intern/cycles/util/math_cdf.h` and `math_cdf.cpp`), and every camera sample reads its raster
//! offset from it (`intern/cycles/kernel/camera/camera.h` `camera_sample`, `kernel/util/lookup_table.h`
//! `lookup_table_read`). Samples are therefore distributed like the filter and carry weight 1.
//! Potter's raster y runs top-down, so `sample` mirrors Cycles' bottom-up y offset.
//!
//! Two Cycles details are kept on purpose, because Blender's pixels depend on them:
//! - the symmetric inversion reads the CDF one cell past the interval that holds each abscissa
//!   (`util_cdf_invert`, `make_symmetric` branch), and
//! - the table's last entry is never written, so it stays 0 and the last cell interpolates towards it.
//!
//! A degenerate CDF cell (zero width) gives `t = 0`. Cycles divides by zero there, which only happens
//! when float32 CDF cells collapse; the f64 table here does not collapse.
use crate::model::PixelFilterType;

/// Cycles' `FILTER_TABLE_SIZE` (`intern/cycles/kernel/types.h`).
const TABLE_SIZE: usize = 1024;

/// Cycles' film filter table for one `(pixel_filter_type, filter_width)` pair.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PixelFilter {
    table: Vec<f64>,
}

impl PixelFilter {
    /// The table `Film::device_update` (`film.cpp`) builds for these film settings. Blender's Box
    /// filter ignores the width (`sync.cpp` sets `filter_width = 1.0` for `FILTER_BOX`).
    pub(crate) fn new(kind: PixelFilterType, width: f64) -> Self {
        Self {
            table: filter_table(kind, width),
        }
    }

    /// The raster offset, in pixels from the pixel's corner, of camera sample `sample_index`.
    ///
    /// `u` and `v` are uniform in `[0, 1)`. Cycles' `integrate_camera_sample` gives sample 0 the
    /// parameter `(0.5, 0.5)`, and `camera_sample` still reads it through the table. So sample 0 is
    /// not exactly the pixel centre: `lookup(0.5)` interpolates between table entries 511 and 512, and
    /// entry 512 sits one step off the centre. At width 1.5 this gives about `(0.50049, 0.49951)` px
    /// for Box and `(0.50053, 0.49947)` px for Blackman-Harris. The y offset is mirrored because
    /// Cycles uses a bottom-up raster while Potter uses a top-down raster. Offsets may leave the pixel.
    pub(crate) fn sample(&self, sample_index: u32, u: f64, v: f64) -> (f64, f64) {
        let (u, v) = if sample_index == 0 {
            (0.5, 0.5)
        } else {
            (u, v)
        };
        (self.lookup(u), 1.0 - self.lookup(v))
    }

    /// Cycles' `lookup_table_read` for the filter table: linear interpolation over `u` in `[0, 1]`.
    fn lookup(&self, u: f64) -> f64 {
        let last = TABLE_SIZE - 1;
        let x = u.clamp(0.0, 1.0) * last as f64;
        let index = (x.floor() as usize).min(last);
        let next = (index + 1).min(last);
        let t = x - index as f64;
        let first = self.table[index];
        if t == 0.0 {
            return first;
        }
        (1.0 - t) * first + t * self.table[next]
    }
}

/// The filter's value at distance `v` from its centre, for the width Cycles uses (`film.cpp`).
fn filter_value(kind: PixelFilterType, v: f64, width: f64) -> f64 {
    match kind {
        PixelFilterType::Box => 1.0,
        PixelFilterType::Gaussian => {
            let v = v * 6.0 / width;
            (-2.0 * v * v).exp()
        }
        PixelFilterType::BlackmanHarris => {
            let v = std::f64::consts::TAU * (v / width + 0.5);
            0.35875 - 0.48829 * v.cos() + 0.14128 * (2.0 * v).cos() - 0.01168 * (3.0 * v).cos()
        }
    }
}

/// Cycles' `filter_table` (`film.cpp`): the inverse CDF of the filter over its symmetric support.
/// The width is scaled as Cycles does (Gaussian x3, Blackman-Harris x2; Box is fixed at 1), and the
/// support is `[-range, range]` with `range = width / 2`.
fn filter_table(kind: PixelFilterType, width: f64) -> Vec<f64> {
    let width = match kind {
        PixelFilterType::Box => 1.0,
        PixelFilterType::Gaussian => width * 3.0,
        PixelFilterType::BlackmanHarris => width * 2.0,
    };
    let range = width * 0.5;
    let cdf = filter_cdf(kind, width, range);
    invert_symmetric_cdf(&cdf, range)
}

/// Cycles' `util_cdf_evaluate` (`math_cdf.h`), at `TABLE_SIZE - 1` abscissae over `[0, range]`:
/// a running sum of `|f|`, normalised, with the last entry forced to 1.
fn filter_cdf(kind: PixelFilterType, width: f64, range: f64) -> Vec<f64> {
    let samples = TABLE_SIZE - 1;
    let mut cdf = vec![0.0; samples + 1];
    for i in 0..samples {
        let x = range * i as f64 / (samples - 1) as f64;
        cdf[i + 1] = cdf[i] + filter_value(kind, x, width).abs();
    }
    let scale = 1.0 / cdf[samples];
    for value in &mut cdf {
        *value *= scale;
    }
    cdf[samples] = 1.0;
    cdf
}

/// Cycles' `util_cdf_invert` with `make_symmetric`: fills `[0, TABLE_SIZE)` from the centre out.
/// Entry `TABLE_SIZE - 1` is never written and stays 0, as in Cycles.
fn invert_symmetric_cdf(cdf: &[f64], range: f64) -> Vec<f64> {
    let cdf_size = cdf.len();
    let mut inverse = vec![0.0; TABLE_SIZE];
    let half_size = (TABLE_SIZE - 1) / 2;
    for i in 0..=half_size {
        let x = i as f64 / half_size as f64;
        // C++ `upper_bound` on the sorted CDF: the count of entries not greater than `x`.
        let mut index = cdf.partition_point(|&value| value <= x);
        let t = if index < cdf_size - 1 {
            let width = cdf[index + 1] - cdf[index];
            if width == 0.0 {
                0.0
            } else {
                (x - cdf[index]) / width
            }
        } else {
            index = cdf_size - 1;
            0.0
        };
        let y = ((index as f64 + t) / (TABLE_SIZE - 1) as f64) * (2.0 * range);
        inverse[half_size + i] = f64::midpoint(1.0, y);
        inverse[half_size - i] = f64::midpoint(1.0, -y);
    }
    inverse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_filter_samples_are_uniform_over_one_pixel() {
        let filter = PixelFilter::new(PixelFilterType::Box, 1.0);
        for step in 0..100 {
            // Avoid the final table entry, which is zero in Cycles.
            let u = f64::from(step) / 100.0;
            let (x, y) = filter.sample(1, u, u);
            assert!((x - 0.5 - (u - 0.5)).abs() < 2.0e-3, "x offset at u = {u}");
            assert!(
                (-0.5..=0.5).contains(&(x - 0.5)),
                "x offset out of box at u = {u}"
            );
            assert!(
                (-0.5..=0.5).contains(&(y - 0.5)),
                "y offset out of box at v = {u}"
            );
            assert!((y - 0.5 - (0.5 - u)).abs() < 2.0e-3, "y offset at v = {u}");
        }
    }

    #[test]
    fn blackman_harris_offsets_cover_the_filter_support_about_the_pixel_centre() {
        let filter = PixelFilter::new(PixelFilterType::BlackmanHarris, 1.5);
        let (low, high) = (filter.lookup(0.0), filter.lookup(1.0));
        assert!((low - (0.5 - 1.5)).abs() < 1.0e-9, "low end {low}");
        // The last table entry is 0, so the top of the range interpolates down to it.
        assert!(
            filter.lookup(0.9999) > 0.0 && high == 0.0,
            "high end {high}"
        );
        assert!((filter.lookup(0.5) - 0.5).abs() < 2.0e-3);
    }

    #[test]
    fn table_is_symmetric_about_the_centre_and_has_the_cycles_gap() {
        for kind in [
            PixelFilterType::Box,
            PixelFilterType::Gaussian,
            PixelFilterType::BlackmanHarris,
        ] {
            let filter = PixelFilter::new(kind, 1.5);
            // Entry 511 is written by both mirror halves (`i = 0`), so the pairs stop before it.
            for (index, &value) in filter.table.iter().enumerate().take(511) {
                let mirrored = filter.table[1022 - index];
                assert!(
                    (value + mirrored - 1.0).abs() < 1.0e-9,
                    "{kind:?} entry {index}"
                );
            }
            assert_eq!(filter.table[TABLE_SIZE - 1], 0.0);
        }
    }

    #[test]
    fn first_sample_reads_the_table_at_one_half() {
        let filter = PixelFilter::new(PixelFilterType::Gaussian, 1.5);
        // Cycles passes (0.5, 0.5) through `lookup_table_read` for sample 0, ignoring the randoms.
        // Potter's top-down raster mirrors Cycles' bottom-up y offset.
        let (x, y) = filter.sample(0, 0.01, 0.99);
        assert!((x - filter.lookup(0.5)).abs() < 1.0e-12);
        assert!((y - (1.0 - filter.lookup(0.5))).abs() < 1.0e-12);
        assert!(
            (x - 0.5).abs() > 1.0e-9,
            "sample 0 must not be the exact centre"
        );
        let (x, y) = filter.sample(1, 0.25, 0.75);
        assert!((x - filter.lookup(0.25)).abs() < 1.0e-12);
        assert!((y - (1.0 - filter.lookup(0.75))).abs() < 1.0e-12);
    }
}
