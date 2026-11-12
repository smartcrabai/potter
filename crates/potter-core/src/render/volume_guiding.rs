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

    pub(super) fn record(&mut self, sample: VolumeGuidingSample) {
        self.sample_count = self.sample_count.wrapping_add(1);
        if sample.primary_transmit {
            self.optical_depth_sum += sample.optical_depth;
            self.optical_depth_count += 1.0;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct VolumeGuidingRead {
    pub(super) scatter: Vec3,
    pub(super) transmit: Vec3,
    pub(super) optical_depth: f32,
}

impl Default for VolumeGuidingRead {
    fn default() -> Self {
        Self {
            scatter: Vec3::ZERO,
            transmit: Vec3::ZERO,
            optical_depth: f32::MAX,
        }
    }
}

impl VolumeGuidingRead {
    pub(super) fn homogeneous_scatter_probability(
        &self,
        extinction: DVec3,
        distance: f64,
    ) -> DVec3 {
        let sigma_t = extinction.as_vec3();
        let distance = distance as f32;
        let attenuation = Vec3::ONE - (-sigma_t * distance).exp();
        let volume = self.scatter + self.transmit;
        let guided = if volume == Vec3::ZERO {
            Vec3::new(
                if sigma_t.x > 0.0 { 0.5 } else { 0.0 },
                if sigma_t.y > 0.0 { 0.5 } else { 0.0 },
                if sigma_t.z > 0.0 { 0.5 } else { 0.0 },
            )
        } else {
            let scale = source_reduce_max(attenuation)
                / (1.0 - (-self.optical_depth).exp());
            source_clamp_vec3(source_safe_divide(self.scatter, volume) * scale, 0.0, 1.0)
        };
        (attenuation + 0.75 * (guided - attenuation)).as_dvec3()
    }

    pub(super) fn heterogeneous_parameters(&self) -> (f64, f64) {
        let scattered = reduce_add(self.scatter);
        let transmitted = reduce_add(self.transmit);
        let volume = transmitted + scattered;
        let mut scatter_probability = if volume == 0.0 {
            0.5
        } else {
            (scattered / volume).min(0.9999)
        };
        let mut majorant_scale = if self.optical_depth == 0.0 {
            1.0
        } else {
            -fast_log(1.0 - scatter_probability) / self.optical_depth
        };
        if majorant_scale < 1.0 {
            majorant_scale = 1.0;
            scatter_probability /= 1.0 - fast_exp(-self.optical_depth);
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
        let pixel_count = width
            .checked_mul(height)
            .ok_or_else(|| allocation_error("volume guiding dimensions exceed addressable memory"))?;
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

    pub(super) fn buffers(
        &mut self,
    ) -> (&[VolumeGuidingRead], &mut [VolumeGuidingAccumulator]) {
        (&self.reads, &mut self.accumulators)
    }

    pub(super) fn filter(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let mut scatter = Vec3::ZERO;
                let mut transmit = Vec3::ZERO;
                for (tap, gaussian) in GAUSSIAN.iter().copied().enumerate() {
                    let neighbor_x = x as isize + tap as isize - FILTER_RADIUS as isize;
                    if !(0..self.width as isize).contains(&neighbor_x) {
                        continue;
                    }
                    let index = y * self.width + neighbor_x as usize;
                    let accumulator = self.accumulators[index];
                    let weight = gaussian / accumulator.sample_count as f32;
                    scatter += accumulator.scatter_sum * weight;
                    transmit += accumulator.transmit_sum * weight;
                }
                let index = y * self.width + x;
                let accumulator = self.accumulators[index];
                self.reads[index] = VolumeGuidingRead {
                    scatter: rgbe_round_trip(scatter),
                    transmit: rgbe_round_trip(transmit),
                    optical_depth: if accumulator.optical_depth_count == 0.0 {
                        f32::MAX
                    } else {
                        accumulator.optical_depth_sum / accumulator.optical_depth_count
                    },
                };
            }
        }

        for x in 0..self.width {
            let mut scatter_neighbors = [Vec3::ZERO; 11];
            let mut transmit_neighbors = [Vec3::ZERO; 11];
            for i in 0..FILTER_RADIUS {
                if i < self.height {
                    let read = self.reads[i * self.width + x];
                    scatter_neighbors[i] = read.scatter;
                    transmit_neighbors[i] = read.transmit;
                }
            }

            let mut index = FILTER_RADIUS;
            for y in 0..self.height {
                let next_y = y + FILTER_RADIUS;
                if next_y >= self.height {
                    scatter_neighbors[index] = Vec3::ZERO;
                    transmit_neighbors[index] = Vec3::ZERO;
                } else {
                    let read = self.reads[next_y * self.width + x];
                    scatter_neighbors[index] = read.scatter;
                    transmit_neighbors[index] = read.transmit;
                }

                index = (index + 1) % GAUSSIAN.len();
                let mut scatter = Vec3::ZERO;
                let mut transmit = Vec3::ZERO;
                for tap in 0..GAUSSIAN.len() {
                    scatter += GAUSSIAN[tap] * scatter_neighbors[(index + tap) % GAUSSIAN.len()];
                    transmit += GAUSSIAN[tap] * transmit_neighbors[(index + tap) % GAUSSIAN.len()];
                }
                let pixel = y * self.width + x;
                self.reads[pixel].scatter = rgbe_round_trip(scatter.abs());
                self.reads[pixel].transmit = rgbe_round_trip(transmit.abs());
            }
        }
    }
}

pub(super) fn next_batch_end(completed: u32, total: u32) -> u32 {
    if completed >= total {
        return total;
    }
    if completed == 0 {
        return total.min(1);
    }
    completed.saturating_mul(2).min(total)
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
        if denominator.x != 0.0 { numerator.x / denominator.x } else { 0.0 },
        if denominator.y != 0.0 { numerator.y / denominator.y } else { 0.0 },
        if denominator.z != 0.0 { numerator.z / denominator.z } else { 0.0 },
    )
}

fn rgbe_round_trip(value: Vec3) -> Vec3 {
    decode_rgbe(encode_rgbe(value))
}

fn encode_rgbe(value: Vec3) -> f32 {
    const RGBE_MAX: f32 = 65_280.0;
    const MIN_VALUE: f32 = 1.0 / 16_777_216.0;
    let absolute = value.abs();
    let max_value = source_min(source_max(source_max(absolute.x, absolute.y), absolute.z), RGBE_MAX);
    if max_value < MIN_VALUE {
        return 0.0;
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
    let r = (value.x.abs() * scale).round().min(255.0) as u8;
    let g = (value.y.abs() * scale).round().min(255.0) as u8;
    let b = (value.z.abs() * scale).round().min(255.0) as u8;
    let e = (((exponent + 15) & 0x1f) as u8) | sign_bits as u8;
    f32::from_bits(u32::from_ne_bytes([r, g, b, e]))
}

fn decode_rgbe(value: f32) -> Vec3 {
    if value == 0.0 {
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
    f32::from_bits(result.to_bits().wrapping_add((exponent as u32).wrapping_shl(23)))
}

#[cfg(test)]
mod tests {
    use super::{
        GAUSSIAN, VolumeGuidingAccumulator, VolumeGuidingFilm, VolumeGuidingRead,
        VolumeGuidingSample, decode_rgbe, encode_rgbe, next_batch_end, rgbe_round_trip,
    };
    use glam::{DVec3, Vec3};

    #[test]
    fn rgbe_matches_source_boundaries_and_signed_round_trip() {
        assert_eq!(encode_rgbe(Vec3::ZERO).to_bits(), 0);
        assert_eq!(encode_rgbe(Vec3::splat(2.0_f32.powi(-25))).to_bits(), 0);
        let minimum = encode_rgbe(Vec3::splat(2.0_f32.powi(-24)));
        assert_eq!(minimum.to_ne_bytes(), [1, 1, 1, 0]);
        assert_eq!(decode_rgbe(minimum), Vec3::splat(2.0_f32.powi(-23)));

        let sample = encode_rgbe(Vec3::new(1.0, 0.5, -0.25));
        assert_eq!(sample.to_ne_bytes(), [128, 64, 32, 48]);
        assert_eq!(decode_rgbe(sample), Vec3::new(1.0, 0.5, -0.25));

        let exponent_carry = encode_rgbe(Vec3::new(1.99999, 1.0, 0.0));
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
        film.filter();
        let (reads, _) = film.buffers();
        let after_x = rgbe_round_trip(scatter * GAUSSIAN[5]);
        let after_y = rgbe_round_trip(after_x * GAUSSIAN[5]);
        assert_eq!(reads[0].scatter, after_y);
        let after_x = rgbe_round_trip(transmit * GAUSSIAN[5]);
        let after_y = rgbe_round_trip(after_x * GAUSSIAN[5]);
        assert_eq!(reads[0].transmit, after_y);
        assert_eq!(reads[0].optical_depth, 4.0);
    }

    #[test]
    fn read_probabilities_match_zero_and_positive_guide_branches() {
        let empty = VolumeGuidingRead::default();
        assert_eq!(empty.optical_depth, f32::MAX);
        assert_eq!(empty.heterogeneous_parameters(), (0.5, 1.0));
        let extinction = DVec3::new(1.0, 0.0, 2.0);
        let distance = 0.5;
        let probability = empty.homogeneous_scatter_probability(extinction, distance);
        let attenuation = Vec3::new(
            1.0 - (-0.5_f32).exp(),
            0.0,
            1.0 - (-1.0_f32).exp(),
        );
        let guided = Vec3::new(0.5, 0.0, 0.5);
        let expected = (attenuation + 0.75 * (guided - attenuation)).as_dvec3();
        assert_eq!(probability, expected);

        let guided_read = VolumeGuidingRead {
            scatter: Vec3::new(3.0, 0.0, 0.0),
            transmit: Vec3::new(1.0, 0.0, 0.0),
            optical_depth: 1.0,
        };
        let (probability, scale) = guided_read.heterogeneous_parameters();
        assert_eq!(probability, 1.0);
        assert!((scale - 1.3863).abs() < 0.001);

        let positive_homogeneous =
            guided_read.homogeneous_scatter_probability(extinction, 0.1);
        let attenuation = Vec3::new(
            1.0 - (-0.1_f32).exp(),
            0.0,
            1.0 - (-0.2_f32).exp(),
        );
        let scale = attenuation.z / (1.0 - (-1.0_f32).exp());
        let guided = Vec3::new(0.75 * scale, 0.0, 0.0);
        let expected = (attenuation + 0.75 * (guided - attenuation)).as_dvec3();
        assert_eq!(positive_homogeneous, expected);

        let low_scale = VolumeGuidingRead {
            optical_depth: 10.0,
            ..guided_read
        };
        let (probability, scale) = low_scale.heterogeneous_parameters();
        assert_eq!(scale, 1.0);
        assert!((probability - 0.75).abs() < 0.001);
        let zero_depth = VolumeGuidingRead {
            optical_depth: 0.0,
            ..guided_read
        };
        assert_eq!(zero_depth.heterogeneous_parameters(), (1.0, 1.0));
    }

    #[test]
    fn records_samples_and_primary_optical_depth_only() {
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

        assert_eq!(accumulator.sample_count, 3);
        assert_eq!(accumulator.scatter_sum, Vec3::splat(1.0));
        assert_eq!(accumulator.transmit_sum, Vec3::splat(2.0));
        assert_eq!(accumulator.optical_depth_count, 2.0);
        assert_eq!(accumulator.optical_depth_sum, 10.0);
        let mut rounding = VolumeGuidingAccumulator::default();
        rounding.add_scatter(Vec3::splat(16_777_216.0));
        rounding.add_scatter(Vec3::splat(1.0));
        rounding.add_scatter(Vec3::splat(-16_777_216.0));
        assert_eq!(rounding.scatter_sum, Vec3::ZERO);

        let mut film = VolumeGuidingFilm::new(1, 1).expect("one-pixel film allocation");
        let (_, accumulators) = film.buffers();
        accumulators[0] = accumulator;
        film.filter();
        let (reads, _) = film.buffers();
        assert_eq!(reads[0].optical_depth, 5.0);
    }


    #[test]
    fn source_overflow_values_are_preserved() {
        let mut film = VolumeGuidingFilm::new(1, 1).expect("one-pixel film allocation");
        let (_, accumulators) = film.buffers();
        accumulators[0].record(VolumeGuidingSample {
            optical_depth: f32::MAX,
            primary_transmit: true,
        });
        accumulators[0].record(VolumeGuidingSample {
            optical_depth: f32::MAX,
            primary_transmit: true,
        });
        film.filter();
        let (reads, _) = film.buffers();
        assert!(reads[0].optical_depth.is_infinite());
        assert!(reads[0].optical_depth.is_sign_positive());

        let read = VolumeGuidingRead {
            scatter: Vec3::ONE,
            transmit: Vec3::ZERO,
            optical_depth: f32::INFINITY,
        };
        assert_eq!(
            read.homogeneous_scatter_probability(
                DVec3::splat(2.0),
                f64::from(f32::MAX),
            ),
            DVec3::splat(1.0),
        );
    }

    #[test]
    fn batches_end_at_power_of_two_checkpoints_without_overflow() {
        let mut completed = 0;
        let mut ends = Vec::new();
        while completed < 13 {
            completed = next_batch_end(completed, 13);
            ends.push(completed);
        }
        assert_eq!(ends, [1, 2, 4, 8, 13]);
        assert_eq!(next_batch_end(0, 0), 0);
        assert_eq!(next_batch_end(u32::MAX - 1, u32::MAX), u32::MAX);
        assert_eq!(next_batch_end(u32::MAX, u32::MAX), u32::MAX);
    }
}
