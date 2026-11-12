use glam::{DVec3, Vec3};

use crate::error::{ErrorCode, PotError, Result};

const GAUSSIAN: [f32; 11] = [
    0.0012273699895602,
    0.0084674212370284,
    0.0379843612914121,
    0.1108921888487800,
    0.2108379677336155,
    0.2611813817992076,
    0.2108379677336155,
    0.1108921888487800,
    0.0379843612914121,
    0.0084674212370284,
    0.0012273699895602,
];
const FILTER_RADIUS: usize = 5;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct VolumeGuidingSample {
    pub(super) optical_depth: f32,
    pub(super) primary_transmit: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct VolumeGuidingAccumulator {
    scatter_sum: Vec3,
    transmit_sum: Vec3,
    optical_depth_sum: f32,
    sample_count: u32,
    optical_depth_count: f32,
}

impl VolumeGuidingAccumulator {
    pub(super) fn add_scatter(&mut self, contribution: Vec3) {
        self.scatter_sum += contribution;
    }

    pub(super) fn add_transmit(&mut self, contribution: Vec3) {
        self.transmit_sum += contribution;
    }

    pub(super) fn sample_count(&self) -> u32 {
        self.sample_count
    }
    pub(super) fn record(&mut self, sample: VolumeGuidingSample) {
        self.sample_count = self.sample_count.wrapping_add(1);
        if sample.primary_transmit {
            self.optical_depth_sum += sample.optical_depth;
            self.optical_depth_count += 1.0;
        }
    }

    pub(super) fn mean_optical_depth(&self) -> f32 {
        if self.optical_depth_count == 0.0 {
            f32::MAX
        } else {
            self.optical_depth_sum / self.optical_depth_count
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct VolumeGuidingRead {
    scatter_rgbe: u32,
    transmit_rgbe: u32,
}

impl VolumeGuidingRead {
    pub(super) fn homogeneous_scatter_probability(
        &self,
        extinction: DVec3,
        distance: f64,
        optical_depth: f32,
    ) -> DVec3 {
        let sigma_t = extinction.as_vec3();
        let distance = distance as f32;
        let attenuation = Vec3::ONE - (-sigma_t * distance).exp();
        let scatter = decode_rgbe(self.scatter_rgbe);
        let transmit = decode_rgbe(self.transmit_rgbe);
        let volume = scatter + transmit;
        let guided = if volume == Vec3::ZERO {
            Vec3::new(
                if sigma_t.x > 0.0 { 0.5 } else { 0.0 },
                if sigma_t.y > 0.0 { 0.5 } else { 0.0 },
                if sigma_t.z > 0.0 { 0.5 } else { 0.0 },
            )
        } else {
            let scale = source_reduce_max(attenuation) / (1.0 - (-optical_depth).exp());
            source_clamp_vec3(source_safe_divide(scatter, volume) * scale, 0.0, 1.0)
        };
        (attenuation + 0.75 * (guided - attenuation)).as_dvec3()
    }

    pub(super) fn heterogeneous_parameters(&self, optical_depth: f32) -> (f64, f64) {
        let scattered = reduce_add(decode_rgbe(self.scatter_rgbe));
        let transmitted = reduce_add(decode_rgbe(self.transmit_rgbe));
        let volume = transmitted + scattered;
        let mut scatter_probability = if volume == 0.0 {
            0.5
        } else {
            (scattered / volume).min(0.9999)
        };
        let mut majorant_scale = if optical_depth == 0.0 {
            1.0
        } else {
            -fast_log(1.0 - scatter_probability) / optical_depth
        };
        if majorant_scale < 1.0 {
            majorant_scale = 1.0;
            scatter_probability =
                source_safe_divide_scalar(scatter_probability, 1.0 - fast_exp(-optical_depth));
        } else {
            scatter_probability = 1.0;
        }
        (f64::from(scatter_probability), f64::from(majorant_scale))
    }
}

pub(super) struct VolumeGuidingFilm {
    width: usize,
    height: usize,
    reads: Vec<VolumeGuidingRead>,
    accumulators: Vec<VolumeGuidingAccumulator>,
}

impl VolumeGuidingFilm {
    pub(super) fn new(width: u32, height: u32) -> Result<Self> {
        let width = usize::try_from(width)
            .map_err(|_| allocation_error("volume guiding width exceeds addressable memory"))?;
        let height = usize::try_from(height)
            .map_err(|_| allocation_error("volume guiding height exceeds addressable memory"))?;
        let pixel_count = width.checked_mul(height).ok_or_else(|| {
            allocation_error("volume guiding dimensions exceed addressable memory")
        })?;
        let reads = allocate_filled(
            pixel_count,
            VolumeGuidingRead::default(),
            "volume guiding read buffer allocation failed",
        )?;
        let accumulators = allocate_filled(
            pixel_count,
            VolumeGuidingAccumulator::default(),
            "volume guiding accumulator allocation failed",
        )?;
        Ok(Self {
            width,
            height,
            reads,
            accumulators,
        })
    }

    pub(super) fn buffers(&mut self) -> (&[VolumeGuidingRead], &mut [VolumeGuidingAccumulator]) {
        (&self.reads, &mut self.accumulators)
    }

    pub(super) fn filter(&mut self) -> Result<()> {
        let width = self.width;
        let height = self.height;
        if width == 0 || height == 0 {
            return Ok(());
        }

        let worker_count =
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        filter_x_parallel(
            &self.accumulators,
            &mut self.reads,
            width,
            height,
            worker_count.min(height),
        )?;
        let column_bands =
            split_column_bands(&mut self.reads, width, height, worker_count.min(width))?;
        filter_y_parallel(column_bands)?;
        Ok(())
    }
}

pub(super) fn next_batch_end(completed: u32, total: u32) -> u32 {
    if completed >= total {
        return total;
    }
    let checkpoint = if completed == 0 {
        1
    } else {
        completed
            .saturating_add(1)
            .checked_next_power_of_two()
            .unwrap_or(u32::MAX)
    };
    checkpoint.min(total)
}

fn filter_x_parallel(
    accumulators: &[VolumeGuidingAccumulator],
    reads: &mut [VolumeGuidingRead],
    width: usize,
    height: usize,
    worker_count: usize,
) -> Result<()> {
    if worker_count == 1 {
        filter_x_rows(accumulators, reads, width, 0);
        return Ok(());
    }

    let rows_per_band = height.div_ceil(worker_count);
    let pixels_per_band = rows_per_band * width;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(worker_count)
            .map_err(|_| allocation_error("volume guiding X worker allocation failed"))?;
        for (band_index, rows) in reads.chunks_mut(pixels_per_band).enumerate() {
            let start_y = band_index * rows_per_band;
            handles.push(std::thread::Builder::new().spawn_scoped(scope, move || {
                filter_x_rows(accumulators, rows, width, start_y);
            }));
        }
        join_filter_workers(handles)
    })
}

fn filter_x_rows(
    accumulators: &[VolumeGuidingAccumulator],
    reads: &mut [VolumeGuidingRead],
    width: usize,
    start_y: usize,
) {
    for (local_y, row) in reads.chunks_exact_mut(width).enumerate() {
        let y = start_y + local_y;
        for x in 0..width {
            let mut scatter = Vec3::ZERO;
            let mut transmit = Vec3::ZERO;
            for (tap, gaussian) in GAUSSIAN.iter().copied().enumerate() {
                let neighbor_x = x as isize + tap as isize - FILTER_RADIUS as isize;
                if !(0..width as isize).contains(&neighbor_x) {
                    continue;
                }
                let index = y * width + neighbor_x as usize;
                let accumulator = accumulators[index];
                let weight = gaussian / accumulator.sample_count as f32;
                scatter += accumulator.scatter_sum * weight;
                transmit += accumulator.transmit_sum * weight;
            }
            row[x] = VolumeGuidingRead {
                scatter_rgbe: encode_rgbe(scatter),
                transmit_rgbe: encode_rgbe(transmit),
            };
        }
    }
}

fn split_column_bands<'read>(
    reads: &'read mut [VolumeGuidingRead],
    width: usize,
    height: usize,
    band_count: usize,
) -> Result<Vec<Vec<&'read mut [VolumeGuidingRead]>>> {
    let mut bands = Vec::new();
    bands
        .try_reserve_exact(band_count)
        .map_err(|_| allocation_error("volume guiding column-band allocation failed"))?;
    for _ in 0..band_count {
        let mut rows = Vec::new();
        rows.try_reserve_exact(height)
            .map_err(|_| allocation_error("volume guiding column-band allocation failed"))?;
        bands.push(rows);
    }

    for row in reads.chunks_mut(width) {
        let mut remaining = row;
        for (index, band) in bands.iter_mut().enumerate() {
            let count = band_width(width, band_count, index);
            let (left, rest) = remaining.split_at_mut(count);
            band.push(left);
            remaining = rest;
        }
    }
    Ok(bands)
}

fn band_width(total: usize, band_count: usize, index: usize) -> usize {
    let base = total / band_count;
    let remainder = total % band_count;
    base + if index < remainder { 1 } else { 0 }
}

fn filter_y_parallel<'read>(bands: Vec<Vec<&'read mut [VolumeGuidingRead]>>) -> Result<()> {
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
            .map_err(|_| allocation_error("volume guiding Y worker allocation failed"))?;
        for rows in bands {
            handles.push(std::thread::Builder::new().spawn_scoped(scope, move || {
                filter_y_columns(rows);
            }));
        }
        join_filter_workers(handles)
    })
}

fn filter_y_columns<'read>(mut rows: Vec<&'read mut [VolumeGuidingRead]>) {
    if rows.is_empty() {
        return;
    }
    let height = rows.len();
    let width = rows[0].len();
    for x in 0..width {
        let mut scatter_neighbors = [Vec3::ZERO; 11];
        let mut transmit_neighbors = [Vec3::ZERO; 11];
        for i in 0..FILTER_RADIUS {
            if i < height {
                let read = rows[i][x];
                scatter_neighbors[i] = decode_rgbe(read.scatter_rgbe);
                transmit_neighbors[i] = decode_rgbe(read.transmit_rgbe);
            }
        }

        let mut index = FILTER_RADIUS;
        for y in 0..height {
            let next_y = y + FILTER_RADIUS;
            if next_y >= height {
                scatter_neighbors[index] = Vec3::ZERO;
                transmit_neighbors[index] = Vec3::ZERO;
            } else {
                let read = rows[next_y][x];
                scatter_neighbors[index] = decode_rgbe(read.scatter_rgbe);
                transmit_neighbors[index] = decode_rgbe(read.transmit_rgbe);
            }

            index = (index + 1) % GAUSSIAN.len();
            let mut scatter = Vec3::ZERO;
            let mut transmit = Vec3::ZERO;
            for tap in 0..GAUSSIAN.len() {
                scatter += GAUSSIAN[tap] * scatter_neighbors[(index + tap) % GAUSSIAN.len()];
                transmit += GAUSSIAN[tap] * transmit_neighbors[(index + tap) % GAUSSIAN.len()];
            }
            rows[y][x] = VolumeGuidingRead {
                scatter_rgbe: encode_rgbe(scatter.abs()),
                transmit_rgbe: encode_rgbe(transmit.abs()),
            };
        }
    }
}

fn join_filter_workers<'scope>(
    handles: Vec<std::io::Result<std::thread::ScopedJoinHandle<'scope, ()>>>,
) -> Result<()> {
    let mut first_error = None;
    for handle in handles {
        match handle {
            Ok(handle) => {
                if handle.join().is_err() && first_error.is_none() {
                    first_error = Some(PotError::new(
                        ErrorCode::InternalError,
                        "volume guiding filter worker panicked",
                    ));
                }
            }
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(PotError::new(
                        ErrorCode::InternalError,
                        format!("cannot start a volume guiding filter worker: {error}"),
                    ));
                }
            }
        }
    }
    first_error.map_or(Ok(()), Err)
}
fn allocate_filled<T: Clone>(length: usize, value: T, message: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| allocation_error(message))?;
    values.resize(length, value);
    Ok(values)
}

fn allocation_error(message: &str) -> PotError {
    PotError::new(ErrorCode::LimitExceeded, message)
}

fn reduce_add(value: Vec3) -> f32 {
    (value.x + value.y) + value.z
}

fn source_reduce_max(value: Vec3) -> f32 {
    source_max(source_max(value.x, value.y), value.z)
}

fn source_max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

fn source_min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

fn source_clamp(value: f32, minimum: f32, maximum: f32) -> f32 {
    source_min(source_max(value, minimum), maximum)
}

fn source_clamp_vec3(value: Vec3, minimum: f32, maximum: f32) -> Vec3 {
    Vec3::new(
        source_clamp(value.x, minimum, maximum),
        source_clamp(value.y, minimum, maximum),
        source_clamp(value.z, minimum, maximum),
    )
}

fn source_safe_divide(numerator: Vec3, denominator: Vec3) -> Vec3 {
    Vec3::new(
        if denominator.x != 0.0 {
            numerator.x / denominator.x
        } else {
            0.0
        },
        if denominator.y != 0.0 {
            numerator.y / denominator.y
        } else {
            0.0
        },
        if denominator.z != 0.0 {
            numerator.z / denominator.z
        } else {
            0.0
        },
    )
}

fn source_safe_divide_scalar(numerator: f32, denominator: f32) -> f32 {
    if denominator != 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

#[cfg(test)]
fn rgbe_round_trip(value: Vec3) -> Vec3 {
    decode_rgbe(encode_rgbe(value))
}

fn encode_rgbe(value: Vec3) -> u32 {
    const RGBE_MAX: f32 = 65_280.0;
    const MIN_VALUE: f32 = 1.0 / 16_777_216.0;
    let absolute = value.abs();
    let max_value = source_min(
        source_max(source_max(absolute.x, absolute.y), absolute.z),
        RGBE_MAX,
    );
    if max_value < MIN_VALUE {
        return 0;
    }

    let mut exponent = ((max_value.to_bits() >> 23) as i32 - 127).max(-16) + 1;
    let mut scale = power_of_two(8 - exponent);
    if (max_value * scale).round() as i32 == 256 {
        exponent += 1;
        scale *= 0.5;
    }

    let sign_bits = ((value.x.to_bits() >> 31) << 7)
        | ((value.y.to_bits() >> 31) << 6)
        | ((value.z.to_bits() >> 31) << 5);
    let r = (value.x.abs() * scale).round_ties_even().min(255.0) as u8;
    let g = (value.y.abs() * scale).round_ties_even().min(255.0) as u8;
    let b = (value.z.abs() * scale).round_ties_even().min(255.0) as u8;
    let e = (((exponent + 15) & 0x1f) as u8) | sign_bits as u8;
    u32::from_ne_bytes([r, g, b, e])
}

fn decode_rgbe(value: u32) -> Vec3 {
    if value & 0x7fff_ffff == 0 {
        return Vec3::ZERO;
    }
    let [r, g, b, exponent] = value.to_ne_bytes();
    let scale = power_of_two(i32::from(exponent & 0x1f) - 23);
    Vec3::new(
        with_sign(f32::from(r) * scale, exponent & 0x80 != 0),
        with_sign(f32::from(g) * scale, exponent & 0x40 != 0),
        with_sign(f32::from(b) * scale, exponent & 0x20 != 0),
    )
}

fn with_sign(value: f32, negative: bool) -> f32 {
    f32::from_bits(value.to_bits() | if negative { 0x8000_0000 } else { 0 })
}

fn power_of_two(exponent: i32) -> f32 {
    f32::from_bits(((exponent + 127) as u32) << 23)
}

fn fast_log(value: f32) -> f32 {
    const LN_2: f32 = 0.6931471805599453;
    let value = value.clamp(f32::MIN_POSITIVE, f32::MAX);
    let bits = value.to_bits();
    let exponent = (bits >> 23) as i32 - 127;
    let fraction = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000) - 1.0;
    let fraction2 = fraction * fraction;
    let fraction4 = fraction2 * fraction2;
    let mut high = fraction * -0.00931049621349 + 0.05206469089414;
    let mut low = fraction * 0.47868480909345 - 0.72116591947498;
    high = fraction * high - 0.13753123777116;
    high = fraction * high + 0.24187369696082;
    high = fraction * high - 0.34730547155299;
    low = fraction * low + 1.442689881667200;
    ((fraction4 * high) + (fraction * low) + exponent as f32) * LN_2
}

fn fast_exp(value: f32) -> f32 {
    const LN_2: f32 = 0.6931471805599453;
    let mut value = (value / LN_2).clamp(-126.0, 126.0);
    let exponent = value as i32;
    value -= exponent as f32;
    value = 1.0 - (1.0 - value);
    let mut result = 1.33336498402e-3_f32;
    result = value * result + 9.810352697968e-3;
    result = value * result + 5.551834031939e-2;
    result = value * result + 0.2401793301105;
    result = value * result + 0.693144857883;
    result = value * result + 1.0;
    f32::from_bits(
        result
            .to_bits()
            .wrapping_add((exponent as u32).wrapping_shl(23)),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        GAUSSIAN, VolumeGuidingAccumulator, VolumeGuidingFilm, VolumeGuidingRead,
        VolumeGuidingSample, decode_rgbe, encode_rgbe, next_batch_end, rgbe_round_trip,
    };
    use glam::{DVec3, Vec3};

    fn make_read(scatter: Vec3, transmit: Vec3) -> VolumeGuidingRead {
        VolumeGuidingRead {
            scatter_rgbe: encode_rgbe(scatter),
            transmit_rgbe: encode_rgbe(transmit),
        }
    }

    #[test]
    fn rgbe_matches_source_rounding_boundaries_and_packing() {
        assert_eq!(std::mem::size_of::<VolumeGuidingRead>(), 8);
        assert_eq!(encode_rgbe(Vec3::ZERO), 0);
        assert_eq!(encode_rgbe(Vec3::splat(2.0_f32.powi(-25))), 0);

        let threshold = encode_rgbe(Vec3::splat(2.0_f32.powi(-24)));
        assert_eq!(threshold, 0);
        assert_eq!(decode_rgbe(threshold), Vec3::ZERO);

        let ties = encode_rgbe(Vec3::new(126.5 / 128.0, 127.5 / 128.0, 1.0));
        assert_eq!(ties.to_ne_bytes(), [126, 128, 128, 16]);
        assert_eq!(decode_rgbe(ties), Vec3::new(126.0 / 128.0, 1.0, 1.0));

        let sample = encode_rgbe(Vec3::new(1.0, 0.5, -0.25));
        assert_eq!(sample.to_ne_bytes(), [128, 64, 32, 48]);
        assert_eq!(decode_rgbe(sample), Vec3::new(1.0, 0.5, -0.25));
        assert_eq!(decode_rgbe(0x8000_0000), Vec3::ZERO);

        let exponent_carry = encode_rgbe(Vec3::new(255.5 / 128.0, 1.0, 0.0));
        assert_eq!(exponent_carry.to_ne_bytes(), [128, 64, 0, 17]);
        assert_eq!(decode_rgbe(exponent_carry), Vec3::new(2.0, 1.0, 0.0));

        let maximum = encode_rgbe(Vec3::new(65_280.0, 100_000.0, 0.0));
        assert_eq!(maximum.to_ne_bytes(), [255, 255, 0, 31]);
        assert_eq!(decode_rgbe(maximum), Vec3::new(65_280.0, 65_280.0, 0.0));
    }

    #[test]
    fn filter_preserves_source_border_and_rgbe_intermediate_rounding() {
        let mut film = VolumeGuidingFilm::new(1, 1).expect("one-pixel film allocation");
        let scatter = Vec3::new(0.7, 0.4, 0.2);
        let transmit = Vec3::new(0.15, 0.3, 0.55);
        let sample = VolumeGuidingSample {
            optical_depth: 4.0,
            primary_transmit: true,
        };
        let (_, accumulators) = film.buffers();
        accumulators[0].add_scatter(scatter);
        accumulators[0].add_transmit(transmit);
        accumulators[0].record(sample);
        film.filter().expect("one-pixel filter");
        let (reads, _) = film.buffers();
        let after_x = rgbe_round_trip(scatter * GAUSSIAN[5]);
        let after_y = rgbe_round_trip(after_x * GAUSSIAN[5]);
        assert_eq!(decode_rgbe(reads[0].scatter_rgbe), after_y);
        let after_x = rgbe_round_trip(transmit * GAUSSIAN[5]);
        let after_y = rgbe_round_trip(after_x * GAUSSIAN[5]);
        assert_eq!(decode_rgbe(reads[0].transmit_rgbe), after_y);
    }

    #[test]
    fn probabilities_match_zero_positive_and_tiny_od_branches() {
        let empty = VolumeGuidingRead::default();
        let extinction = DVec3::new(1.0, 0.0, 2.0);
        let distance = 0.5;
        let no_data_od = f32::MAX;
        assert_eq!(empty.heterogeneous_parameters(no_data_od), (0.5, 1.0));
        let probability = empty.homogeneous_scatter_probability(extinction, distance, no_data_od);
        let attenuation = Vec3::new(1.0 - (-0.5_f32).exp(), 0.0, 1.0 - (-1.0_f32).exp());
        let guided = Vec3::new(0.5, 0.0, 0.5);
        let expected = (attenuation + 0.75 * (guided - attenuation)).as_dvec3();
        assert_eq!(probability, expected);

        let guided_read = make_read(Vec3::new(3.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let (probability, scale) = guided_read.heterogeneous_parameters(1.0);
        assert_eq!(probability, 1.0);
        assert!((scale - 1.3863).abs() < 0.001);

        let positive_homogeneous =
            guided_read.homogeneous_scatter_probability(extinction, 0.1, 1.0);
        let attenuation = Vec3::new(1.0 - (-0.1_f32).exp(), 0.0, 1.0 - (-0.2_f32).exp());
        let scale = attenuation.z / (1.0 - (-1.0_f32).exp());
        let guided = Vec3::new(0.75 * scale, 0.0, 0.0);
        let expected = (attenuation + 0.75 * (guided - attenuation)).as_dvec3();
        assert_eq!(positive_homogeneous, expected);

        let (probability, scale) = guided_read.heterogeneous_parameters(10.0);
        assert_eq!(scale, 1.0);
        assert!((probability - 0.75).abs() < 0.001);
        assert_eq!(guided_read.heterogeneous_parameters(0.0), (1.0, 1.0));

        let quantized_zero = make_read(Vec3::splat(2.0_f32.powi(-25)), Vec3::ONE);
        assert_eq!(quantized_zero.scatter_rgbe, 0);
        assert_eq!(
            quantized_zero.heterogeneous_parameters(f32::MIN_POSITIVE),
            (0.0, 1.0)
        );
    }

    #[test]
    fn records_contributions_and_primary_optical_depth_only() {
        let mut accumulator = VolumeGuidingAccumulator::default();
        accumulator.add_scatter(Vec3::splat(1.0));
        accumulator.record(VolumeGuidingSample {
            optical_depth: 100.0,
            primary_transmit: false,
        });
        accumulator.add_transmit(Vec3::splat(2.0));
        accumulator.record(VolumeGuidingSample {
            optical_depth: 2.0,
            primary_transmit: true,
        });
        accumulator.record(VolumeGuidingSample {
            optical_depth: 8.0,
            primary_transmit: true,
        });

        assert_eq!(accumulator.sample_count(), 3);
        assert_eq!(accumulator.scatter_sum, Vec3::splat(1.0));
        assert_eq!(accumulator.transmit_sum, Vec3::splat(2.0));
        assert_eq!(accumulator.optical_depth_count, 2.0);
        assert_eq!(accumulator.optical_depth_sum, 10.0);
        assert_eq!(accumulator.mean_optical_depth(), 5.0);

        let mut rounding = VolumeGuidingAccumulator::default();
        rounding.add_scatter(Vec3::splat(16_777_216.0));
        rounding.add_scatter(Vec3::splat(1.0));
        rounding.add_scatter(Vec3::splat(-16_777_216.0));
        assert_eq!(rounding.scatter_sum, Vec3::ZERO);
    }

    #[test]
    fn mean_optical_depth_changes_between_samples_without_filtering_guides() {
        let mut accumulator = VolumeGuidingAccumulator::default();
        let frozen = make_read(Vec3::splat(0.25), Vec3::splat(0.75));
        let optical_depth_before = accumulator.mean_optical_depth();
        assert_eq!(optical_depth_before, f32::MAX);
        let probability_before =
            frozen.homogeneous_scatter_probability(DVec3::splat(0.1), 0.1, optical_depth_before);

        accumulator.record(VolumeGuidingSample {
            optical_depth: 1.0,
            primary_transmit: true,
        });
        let optical_depth_after = accumulator.mean_optical_depth();
        assert_eq!(optical_depth_after, 1.0);
        assert_eq!(frozen.scatter_rgbe, encode_rgbe(Vec3::splat(0.25)));
        assert_eq!(frozen.transmit_rgbe, encode_rgbe(Vec3::splat(0.75)));
        let probability_after =
            frozen.homogeneous_scatter_probability(DVec3::splat(0.1), 0.1, optical_depth_after);
        assert_ne!(probability_before, probability_after);
    }

    #[test]
    fn source_optical_depth_overflow_is_preserved() {
        let mut accumulator = VolumeGuidingAccumulator::default();
        accumulator.record(VolumeGuidingSample {
            optical_depth: f32::MAX,
            primary_transmit: true,
        });
        accumulator.record(VolumeGuidingSample {
            optical_depth: f32::MAX,
            primary_transmit: true,
        });
        let optical_depth = accumulator.mean_optical_depth();
        assert!(optical_depth.is_infinite());
        assert!(optical_depth.is_sign_positive());

        let read = make_read(Vec3::ONE, Vec3::ZERO);
        assert_eq!(
            read.homogeneous_scatter_probability(
                DVec3::splat(2.0),
                f64::from(f32::MAX),
                optical_depth,
            ),
            DVec3::splat(1.0),
        );
    }

    #[test]
    fn batches_end_at_strict_power_of_two_checkpoints_without_overflow() {
        let mut completed = 0;
        let mut ends = Vec::new();
        while completed < 13 {
            completed = next_batch_end(completed, 13);
            ends.push(completed);
        }
        assert_eq!(ends, [1, 2, 4, 8, 13]);
        assert_eq!(next_batch_end(0, 0), 0);
        assert_eq!(next_batch_end(3, 200), 4);
        assert_eq!(next_batch_end(64, 200), 128);
        assert_eq!(next_batch_end(80, 200), 128);
        assert_eq!(next_batch_end(96, 200), 128);
        assert_eq!(next_batch_end(80, 100), 100);
        assert_eq!(next_batch_end(u32::MAX - 1, u32::MAX), u32::MAX);
        assert_eq!(next_batch_end(u32::MAX, u32::MAX), u32::MAX);
    }
}
