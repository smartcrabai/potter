use glam::DVec3;

use crate::error::{PotError, Result};
use crate::render::world_volume::{volume_rng_scramble, volume_uniform_1d};

use super::{MediumOutcome, Pcg32};

const MAX_EVENTS: usize = 1025;
const VOLUME_RUSSIAN_ROULETTE_THRESHOLD: f64 = 0.05;
const SOURCE_VOLUME_EXPANSION_DIMENSION: u32 = 6;
const SOURCE_VOLUME_SHADE_DIMENSION: u32 = 7;

#[derive(Clone, Copy)]
pub(super) struct Coefficients {
    pub extinction: DVec3,
    pub scatter: DVec3,
    pub emission: DVec3,
}

#[derive(Clone, Copy)]
pub(super) struct Interval {
    pub end: f64,
    pub minorant: f64,
    pub majorant: f64,
}

pub(super) trait Medium {
    fn interval(
        &mut self,
        start: f64,
        limit: f64,
        rng: &mut Pcg32,
        source_rng_offset: u32,
    ) -> Result<Option<Interval>>;
    fn coefficients(&mut self, distance: f64) -> Result<Coefficients>;
    fn extinction(&mut self, distance: f64) -> Result<DVec3>;
}

fn next_interval<M: Medium>(
    medium: &mut M,
    start: f64,
    limit: f64,
    rng: &mut Pcg32,
    source_rng_offset: u32,
) -> Result<Option<Interval>> {
    let Some(mut interval) = medium.interval(start, limit, rng, source_rng_offset)? else {
        return Ok(None);
    };
    interval.end = interval.end.min(limit);
    if interval.end.partial_cmp(&start) != Some(std::cmp::Ordering::Greater) {
        return Err(PotError::invalid_operation(
            "heterogeneous medium interval did not advance the ray",
        ));
    }
    if !interval.majorant.is_finite() || interval.majorant < 0.0 {
        return Err(PotError::invalid_operation(
            "heterogeneous medium returned an invalid majorant",
        ));
    }
    if !interval.minorant.is_finite() || interval.minorant > interval.majorant {
        return Err(PotError::invalid_operation(
            "heterogeneous medium returned an invalid minorant",
        ));
    }
    Ok(Some(interval))
}

#[derive(Clone, Copy)]
pub(super) struct GuidingParameters {
    pub scatter_probability: f64,
    pub majorant_scale: f64,
}

#[derive(Clone, Copy)]
pub(super) struct Settings {
    pub limit: f64,
    pub events: bool,
    pub use_mis: bool,
    pub source_rng_offset: u32,
    pub source_index: u32,
    pub guiding: Option<GuidingParameters>,
}

#[derive(Clone, Copy)]
pub(super) struct ScatterCandidate {
    pub distance: f64,
    pub throughput: DVec3,
    pub distance_pdf: f64,
}

pub(super) struct DirectSample<T> {
    pub distance: f64,
    pub pdf: f64,
    pub payload: T,
    pub throughput: DVec3,
    pub distance_pdf: f64,
    pub processed: bool,
}

pub(super) struct Segment {
    pub outcome: MediumOutcome,
    pub emission: DVec3,
    pub scatter_pdf: f64,
    pub optical_depth: f32,
    pub direct_candidate: Option<ScatterCandidate>,
}

#[derive(Clone, Copy)]
struct ReservoirCandidate {
    scatter: ScatterCandidate,
    emission: DVec3,
}

struct Reservoir {
    total_weight: f64,
    random: f64,
    candidate: Option<ReservoirCandidate>,
}

impl Reservoir {
    fn new(random: f64) -> Self {
        Self {
            total_weight: 0.0,
            random,
            candidate: None,
        }
    }

    fn add_sample(&mut self, weight: f64, candidate: ReservoirCandidate) {
        if weight.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return;
        }

        self.total_weight += weight;
        let threshold = weight / self.total_weight;
        if self.random <= threshold
            || self.total_weight.partial_cmp(&weight) == Some(std::cmp::Ordering::Equal)
        {
            self.candidate = Some(candidate);
            self.random /= threshold;
        } else {
            self.random = (self.random - threshold) / (1.0 - threshold);
        }
        self.random = self.random.clamp(0.0, 1.0);
    }
}

fn channel_pdf(scatter: DVec3, extinction: DVec3, throughput: DVec3) -> DVec3 {
    let albedo = DVec3::from_array(std::array::from_fn(|channel| {
        if extinction[channel] > 0.0 {
            scatter[channel] / extinction[channel]
        } else {
            1.0
        }
    }));
    let weights = (albedo * throughput).abs();
    let sum = weights.element_sum();
    if sum > 0.0 {
        weights / sum
    } else {
        DVec3::splat(1.0 / 3.0)
    }
}

fn scatter_probability(scatter: DVec3, extinction: DVec3, null: DVec3, throughput: DVec3) -> f64 {
    let channels = channel_pdf(scatter, extinction, throughput);
    channels
        .dot(DVec3::from_array(std::array::from_fn(|channel| {
            let proposal = scatter[channel] + null[channel];
            if proposal > 0.0 {
                scatter[channel] / proposal
            } else {
                0.0
            }
        })))
        .clamp(0.0, 1.0)
}

fn event_correction(sampled_majorant: f64, actual_majorant: f64, distance: f64) -> f64 {
    if actual_majorant > sampled_majorant {
        ((sampled_majorant - actual_majorant) * distance).exp() / sampled_majorant
    } else {
        1.0 / sampled_majorant
    }
}

fn equiangular_correction(sampled_majorant: f64, actual_majorant: f64, distance: f64) -> f64 {
    if actual_majorant > sampled_majorant {
        ((sampled_majorant - actual_majorant) * distance).exp()
    } else {
        1.0
    }
}

fn event_majorant(rate: f64, coefficients: Coefficients) -> f64 {
    rate.max(coefficients.extinction.max_element())
}

// Source RNG dimensions are shared across all leaf intervals in one event advance.
#[derive(Default)]
struct TelescopeSamples {
    expansion_sample: Option<f64>,
    shade_offset: Option<f64>,
}

impl TelescopeSamples {
    fn expansion_sample(&mut self, sample: u32, source_rng_offset: u32) -> f64 {
        *self.expansion_sample.get_or_insert_with(|| {
            f64::from(volume_uniform_1d(
                sample,
                source_rng_offset.wrapping_add(SOURCE_VOLUME_EXPANSION_DIMENSION),
            ))
        })
    }

    fn shade_offset(&mut self, source_index: u32, source_rng_offset: u32) -> f64 {
        *self.shade_offset.get_or_insert_with(|| {
            f64::from(volume_uniform_1d(
                source_index,
                source_rng_offset.wrapping_add(SOURCE_VOLUME_SHADE_DIMENSION),
            ))
        })
    }
}

fn transmittance_with_samples<M: Medium>(
    medium: &mut M,
    start: f64,
    end: f64,
    sigma_range: f64,
    source_rng_offset: u32,
    source_index: u32,
    samples: &mut TelescopeSamples,
    rng: &mut Pcg32,
) -> Result<DVec3> {
    const EXPANSION_SURVIVAL: f64 = 0.1;
    const EXPANSION_CUTOFF: u32 = 4;
    const MAX_SAMPLES: f64 = 1024.0;

    let length = end - start;
    let expected_steps = sigma_range * length;
    let samples_per_estimator = expected_steps.round().clamp(1.0, MAX_SAMPLES) as usize;
    let (estimators, pmf) = if sigma_range == 0.0 {
        (1, 1.0)
    } else {
        let expansion_sample = samples.expansion_sample(rng.sample_index, source_rng_offset);
        let order = (expansion_sample.ln() / EXPANSION_SURVIVAL.ln())
            .floor()
            .clamp(0.0, f64::from(EXPANSION_CUTOFF)) as u32;
        let probability = if order == EXPANSION_CUTOFF {
            EXPANSION_SURVIVAL.powi(order as i32)
        } else {
            (1.0 - EXPANSION_SURVIVAL) * EXPANSION_SURVIVAL.powi(order as i32)
        };
        (1_usize << order, probability)
    };

    let sample_count = estimators * samples_per_estimator;
    let shade_offset = samples.shade_offset(source_index, source_rng_offset);
    let step_size = length / sample_count as f64;
    if estimators == 1 {
        let mut optical_depth = DVec3::ZERO;
        for index in 0..samples_per_estimator {
            let distance = (start + (shade_offset + index as f64) * step_size).min(end);
            optical_depth += medium.extinction(distance)?;
        }
        let optical_depth = optical_depth * step_size;
        return Ok(if optical_depth == DVec3::ZERO {
            DVec3::ONE
        } else {
            (-optical_depth).exp()
        });
    }

    let mut optical_depth_by_parity = [DVec3::ZERO; 2];
    let mut optical_depth_j1 = DVec3::ZERO;
    let mut transmittance_k = DVec3::ZERO;
    for estimator in 0..estimators {
        let mut optical_depth_k = DVec3::ZERO;
        for index in 0..samples_per_estimator {
            let sample_index = index * estimators + estimator;
            let distance = (start + (shade_offset + sample_index as f64) * step_size).min(end);
            let optical_depth = medium.extinction(distance)? * step_size;
            optical_depth_k += optical_depth * estimators as f64;
            optical_depth_by_parity[sample_index % 2] += optical_depth * 2.0;
            optical_depth_j1 += optical_depth;
        }
        transmittance_k += (-optical_depth_k).exp();
    }

    let transmittance_j1 = (-optical_depth_j1).exp();
    let transmittance_j =
        ((-optical_depth_by_parity[0]).exp() + (-optical_depth_by_parity[1]).exp()) * 0.5;
    Ok(transmittance_k / estimators as f64 + (transmittance_j1 - transmittance_j) / pmf)
}

pub(super) fn transmittance<M: Medium>(
    medium: &mut M,
    start: f64,
    end: f64,
    sigma_range: f64,
    rng: &mut Pcg32,
    source_rng_offset: u32,
    source_index: u32,
) -> Result<DVec3> {
    transmittance_with_samples(
        medium,
        start,
        end,
        sigma_range,
        source_rng_offset,
        source_index,
        &mut TelescopeSamples::default(),
        rng,
    )
}

fn transmission_event_weight(
    sampled_majorant: f64,
    actual_majorant: f64,
    extinction: DVec3,
    distance: f64,
) -> DVec3 {
    (DVec3::splat(actual_majorant) - extinction).max(DVec3::ZERO)
        * event_correction(sampled_majorant, actual_majorant, distance)
}
fn extended_distance_pdf(null_survival: f64, majorant: f64, scatter_probability: f64) -> f64 {
    null_survival * majorant * scatter_probability
}

fn blend_emission(full: DVec3, candidate: DVec3, weight: f64) -> DVec3 {
    full * (1.0 - weight) + candidate * weight
}

fn finalize_guided_reservoir(
    reservoir: &mut Reservoir,
    transmitted_throughput: DVec3,
    full_emission: DVec3,
    target_scatter_probability: f64,
) -> Option<(ReservoirCandidate, f64, f64, DVec3)> {
    let unguided_probability = reservoir.total_weight;
    if unguided_probability.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    let scatter_emission = reservoir.candidate?.emission;

    let guided_probability = if transmitted_throughput == DVec3::ZERO {
        1.0
    } else {
        0.25 * unguided_probability + 0.75 * target_scatter_probability
    };
    if transmitted_throughput != DVec3::ZERO {
        reservoir.total_weight = guided_probability;
        reservoir.add_sample(
            1.0 - guided_probability,
            ReservoirCandidate {
                scatter: ScatterCandidate {
                    distance: 0.0,
                    throughput: transmitted_throughput,
                    distance_pdf: 0.0,
                },
                emission: full_emission,
            },
        );
    }

    let candidate = reservoir.candidate?;
    let is_scatter = candidate.scatter.distance_pdf > 0.0;
    let scale = if is_scatter {
        unguided_probability / guided_probability
    } else {
        (1.0 - unguided_probability) / (1.0 - guided_probability)
    };
    Some((candidate, scale, unguided_probability, scatter_emission))
}

fn zero_direct_samples<T>(samples: &mut [Option<DirectSample<T>>]) {
    for sample in samples.iter_mut().flatten() {
        sample.throughput = DVec3::ZERO;
        sample.distance_pdf = 0.0;
        sample.processed = true;
    }
}

fn roulette<T>(
    throughput: &mut DVec3,
    reservoir: &mut Reservoir,
    events: bool,
    use_mis: bool,
    direct_samples: &mut [Option<DirectSample<T>>],
    direct_rr_scale: &mut f64,
) -> bool {
    let threshold = throughput.abs().max_element();
    if threshold > VOLUME_RUSSIAN_ROULETTE_THRESHOLD {
        return false;
    }

    let equiangular_pending = use_mis
        && direct_samples
            .iter()
            .flatten()
            .any(|sample| !sample.processed);
    let has_scatter_samples = reservoir.candidate.is_some() && !equiangular_pending;
    if !events || has_scatter_samples {
        if threshold == 0.0 || reservoir.random > threshold {
            *throughput = DVec3::ZERO;
            if equiangular_pending {
                *direct_rr_scale = 0.0;
                zero_direct_samples(direct_samples);
            }
            return true;
        }
        reservoir.random = (reservoir.random / threshold).clamp(0.0, 1.0);
        *throughput /= threshold;
    }

    if equiangular_pending {
        if threshold == 0.0 || reservoir.random > threshold {
            *direct_rr_scale = 0.0;
            zero_direct_samples(direct_samples);
            reservoir.random = (reservoir.random - threshold) / (1.0 - threshold);
        } else {
            reservoir.random = (reservoir.random / threshold).clamp(0.0, 1.0);
            *direct_rr_scale /= threshold;
        }
        reservoir.random = reservoir.random.clamp(0.0, 1.0);
    }
    false
}

fn process_direct_samples<M: Medium, T>(
    medium: &mut M,
    samples: &mut [Option<DirectSample<T>>],
    end: f64,
    interval_start: f64,
    sampled_majorant: f64,
    events: bool,
    throughput: DVec3,
    null_survival: f64,
    direct_rr_scale: f64,
) -> Result<()> {
    for sample in samples.iter_mut().flatten() {
        if sample.processed || sample.distance < interval_start || sample.distance > end {
            continue;
        }

        if sampled_majorant == 0.0 {
            sample.throughput = DVec3::ZERO;
            sample.distance_pdf = 0.0;
            sample.processed = true;
            continue;
        }

        let coefficients = medium.coefficients(sample.distance)?;
        let actual_majorant = event_majorant(sampled_majorant, coefficients);
        let distance = sample.distance - interval_start;
        let correction = equiangular_correction(sampled_majorant, actual_majorant, distance);
        let direct_throughput = if events {
            throughput * (null_survival * correction * direct_rr_scale)
        } else {
            throughput
        };
        let null = DVec3::splat(actual_majorant) - coefficients.extinction;
        let probability = scatter_probability(
            coefficients.scatter,
            coefficients.extinction,
            null,
            direct_throughput,
        );
        sample.throughput = direct_throughput;
        sample.distance_pdf = if events {
            extended_distance_pdf(null_survival, sampled_majorant, probability)
        } else {
            0.0
        };
        sample.processed = true;
    }
    Ok(())
}

fn process_equiangular_transmittance<M: Medium, T>(
    medium: &mut M,
    samples: &mut [Option<DirectSample<T>>],
    start: f64,
    interval: Interval,
    source_rng_offset: u32,
    source_index: u32,
    rng: &mut Pcg32,
    telescope_samples: &mut TelescopeSamples,
) -> Result<()> {
    let sigma_range = (interval.majorant - interval.minorant).max(0.0);
    for sample in samples.iter_mut().flatten() {
        if sample.processed || sample.distance < start {
            continue;
        }

        let sample_end = sample.distance.min(interval.end);
        sample.throughput *= transmittance_with_samples(
            medium,
            start,
            sample_end,
            sigma_range,
            source_rng_offset,
            source_index,
            telescope_samples,
            rng,
        )?;
        sample.distance_pdf = 0.0;
        sample.processed = sample.distance <= interval.end;
    }
    Ok(())
}

pub(super) fn trace<M: Medium, T>(
    medium: &mut M,
    settings: Settings,
    direct_samples: &mut [Option<DirectSample<T>>],
    throughput: &mut DVec3,
    rng: &mut Pcg32,
) -> Result<Segment> {
    let initial_throughput = *throughput;
    let mut path_throughput = initial_throughput;
    let mut emission = DVec3::ZERO;
    let mut null_survival = 1.0;
    let mut optical_depth = 0.0_f32;
    let mut position = 0.0;
    let mut active_interval = if position < settings.limit {
        next_interval(
            medium,
            position,
            settings.limit,
            rng,
            settings.source_rng_offset,
        )?
    } else {
        None
    };
    let Some(interval) = active_interval else {
        return Ok(Segment {
            outcome: MediumOutcome::Survive,
            emission,
            scatter_pdf: 0.0,
            optical_depth,
            direct_candidate: None,
        });
    };
    optical_depth += (interval.majorant as f32) * ((interval.end - position) as f32);
    let mut reservoir = Reservoir::new(f64::from(volume_uniform_1d(
        settings.source_index,
        settings.source_rng_offset.wrapping_add(4),
    )));
    let mut source_rng_offset = volume_rng_scramble(settings.source_rng_offset, 0xe35f_ad82);
    let mut direct_rr_scale = 1.0;
    let mut telescope_samples = TelescopeSamples::default();
    if !settings.use_mis {
        for sample in direct_samples.iter_mut().flatten() {
            if !sample.processed {
                sample.throughput = initial_throughput;
                sample.distance_pdf = 0.0;
            }
        }
        process_equiangular_transmittance(
            medium,
            direct_samples,
            position,
            interval,
            source_rng_offset,
            settings.source_index,
            rng,
            &mut telescope_samples,
        )?;
    }
    let mut event_steps = 0;

    while position < settings.limit && event_steps < MAX_EVENTS {
        event_steps += 1;
        let mut residual_sample = f64::from(volume_uniform_1d(
            settings.source_index,
            source_rng_offset.wrapping_add(5),
        ));
        let mut terminated = false;
        let mut event_found = false;

        loop {
            let interval = if let Some(interval) = active_interval {
                interval
            } else {
                let Some(interval) =
                    next_interval(medium, position, settings.limit, rng, source_rng_offset)?
                else {
                    position = settings.limit;
                    break;
                };
                optical_depth += (interval.majorant as f32) * ((interval.end - position) as f32);
                if !settings.use_mis {
                    process_equiangular_transmittance(
                        medium,
                        direct_samples,
                        position,
                        interval,
                        source_rng_offset,
                        settings.source_index,
                        rng,
                        &mut telescope_samples,
                    )?;
                }
                active_interval = Some(interval);
                interval
            };

            let end = interval.end;
            let rate = interval.majorant
                * settings
                    .guiding
                    .map_or(1.0, |guiding| guiding.majorant_scale);
            if !rate.is_finite() || rate < 0.0 {
                return Err(PotError::invalid_operation(
                    "heterogeneous medium returned an invalid scaled majorant",
                ));
            }
            let candidate_distance = if rate > 0.0 {
                position + -(-residual_sample).ln_1p() / rate
            } else {
                f64::INFINITY
            };

            if candidate_distance <= end {
                if settings.use_mis {
                    process_direct_samples(
                        medium,
                        direct_samples,
                        candidate_distance,
                        position,
                        rate,
                        settings.events,
                        path_throughput,
                        null_survival,
                        direct_rr_scale,
                    )?;
                }
                if roulette(
                    &mut path_throughput,
                    &mut reservoir,
                    settings.events,
                    settings.use_mis,
                    direct_samples,
                    &mut direct_rr_scale,
                ) {
                    terminated = true;
                    break;
                }

                let distance = candidate_distance - position;
                let coefficients = medium.coefficients(candidate_distance)?;
                let actual_majorant = event_majorant(rate, coefficients);
                let correction = event_correction(rate, actual_majorant, distance);
                if settings.guiding.is_some() || reservoir.candidate.is_none() {
                    emission += path_throughput * (coefficients.emission * correction);
                }

                if settings.events {
                    let null = DVec3::splat(actual_majorant) - coefficients.extinction;
                    let probability = scatter_probability(
                        coefficients.scatter,
                        coefficients.extinction,
                        null,
                        path_throughput,
                    );
                    let distance_pdf = extended_distance_pdf(null_survival, rate, probability);
                    let candidate_throughput = if probability > 0.0 {
                        path_throughput * coefficients.scatter * (correction / probability)
                    } else {
                        DVec3::ZERO
                    };
                    let scatter_candidate = ReservoirCandidate {
                        scatter: ScatterCandidate {
                            distance: candidate_distance,
                            throughput: candidate_throughput,
                            distance_pdf,
                        },
                        emission,
                    };
                    if settings.guiding.is_some() {
                        reservoir.add_sample(null_survival * probability, scatter_candidate);
                    } else if reservoir.candidate.is_none() && probability > 0.0 {
                        if reservoir.random <= probability {
                            reservoir.random = (reservoir.random / probability).clamp(0.0, 1.0);
                            reservoir.add_sample(null_survival * probability, scatter_candidate);
                        } else {
                            reservoir.random = ((reservoir.random - probability)
                                / (1.0 - probability))
                                .clamp(0.0, 1.0);
                        }
                    }

                    let null_probability = 1.0 - probability;
                    if null_probability > 0.0 {
                        path_throughput *= null * (correction / null_probability);
                        null_survival *= null_probability;
                    } else {
                        path_throughput = DVec3::ZERO;
                        null_survival = 0.0;
                    }
                } else {
                    path_throughput *= transmission_event_weight(
                        rate,
                        actual_majorant,
                        coefficients.extinction,
                        distance,
                    );
                }

                position = candidate_distance;
                if position >= end {
                    active_interval = None;
                }
                event_found = true;
                break;
            }

            if settings.use_mis {
                process_direct_samples(
                    medium,
                    direct_samples,
                    end,
                    position,
                    rate,
                    settings.events,
                    path_throughput,
                    null_survival,
                    direct_rr_scale,
                )?;
            }
            let interval_length = end - position;
            residual_sample =
                (1.0 - (1.0 - residual_sample) * (rate * interval_length).exp()).clamp(0.0, 1.0);
            position = end;
            active_interval = None;
            if position >= settings.limit {
                break;
            }
        }

        if terminated || !event_found || path_throughput == DVec3::ZERO && !settings.events {
            break;
        }
        if settings.guiding.is_none()
            && reservoir.candidate.is_some()
            && direct_samples.iter().any(Option::is_some)
            && direct_samples
                .iter()
                .flatten()
                .all(|sample| sample.processed)
        {
            break;
        }
        source_rng_offset = source_rng_offset.wrapping_add(super::CYCLES_PRNG_BOUNCE_NUM);
        telescope_samples = TelescopeSamples::default();
    }

    let unguided_reservoir_weight = reservoir.total_weight;
    let direct_candidate = reservoir.candidate.map(|candidate| {
        let weight = if settings.guiding.is_some() {
            unguided_reservoir_weight
        } else {
            1.0
        };
        ScatterCandidate {
            distance: candidate.scatter.distance,
            throughput: candidate.scatter.throughput * weight,
            distance_pdf: candidate.scatter.distance_pdf,
        }
    });

    let mut outcome = MediumOutcome::Survive;
    let mut scatter_pdf = 0.0;
    let mut final_emission = emission;
    if let Some(guiding) = settings.guiding.filter(|_| settings.events) {
        if let Some((candidate, scale, weight, scatter_emission)) = finalize_guided_reservoir(
            &mut reservoir,
            path_throughput,
            emission,
            guiding.scatter_probability,
        ) {
            path_throughput = candidate.scatter.throughput * scale;
            final_emission = blend_emission(emission, scatter_emission, weight);
            if candidate.scatter.distance_pdf > 0.0 {
                outcome = MediumOutcome::Scatter(candidate.scatter.distance);
                scatter_pdf = candidate.scatter.distance_pdf;
            }
        }
    } else if settings.events
        && let Some(candidate) = reservoir.candidate
    {
        path_throughput = candidate.scatter.throughput;
        final_emission = candidate.emission;
        outcome = MediumOutcome::Scatter(candidate.scatter.distance);
        scatter_pdf = candidate.scatter.distance_pdf;
    }

    *throughput = path_throughput;
    Ok(Segment {
        outcome,
        emission: final_emission,
        scatter_pdf,
        optical_depth,
        direct_candidate,
    })
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::super::{MediumOutcome, Pcg32};
    use super::{
        Coefficients, DirectSample, GuidingParameters, Interval, Medium, Reservoir,
        ReservoirCandidate, ScatterCandidate, Settings, TelescopeSamples, blend_emission,
        equiangular_correction, event_correction, event_majorant, extended_distance_pdf,
        finalize_guided_reservoir, process_direct_samples, scatter_probability, trace,
        transmission_event_weight, transmittance, transmittance_with_samples,
    };
    use crate::error::{PotError, Result};
    use crate::render::world_volume::{volume_rng_scramble, volume_uniform_1d};

    #[test]
    fn underestimated_majorant_correction_retains_inverse_rate_units() {
        let correction = event_correction(2.0, 3.0, 0.5);
        assert!((correction - (-0.5_f64).exp() / 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn equiangular_correction_has_no_inverse_majorant_factor() {
        assert!((equiangular_correction(2.0, 3.0, 0.5) - (-0.5_f64).exp()).abs() < 1.0e-12);
        assert_eq!(equiangular_correction(2.0, 2.0, 0.5), 1.0);
    }

    #[test]
    fn source_event_majorant_does_not_promote_emission() {
        assert_eq!(
            event_majorant(
                2.0,
                Coefficients {
                    extinction: DVec3::splat(0.5),
                    scatter: DVec3::splat(0.25),
                    emission: DVec3::splat(100.0),
                },
            ),
            2.0,
        );
    }

    #[test]
    fn scatter_proposal_uses_scatter_and_null_but_not_absorption_events() {
        let probability =
            scatter_probability(DVec3::splat(0.5), DVec3::ONE, DVec3::ONE, DVec3::ONE);
        assert!((probability - 1.0 / 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn terminal_null_event_is_the_majorant_corrected_transmittance_weight() {
        let weight = transmission_event_weight(2.0, 2.0, DVec3::splat(0.5), 0.25);
        assert_eq!(weight, DVec3::splat(0.75));
    }

    #[test]
    fn equiangular_pdf_uses_the_null_extended_same_walk_density() {
        assert_eq!(extended_distance_pdf(0.25, 4.0, 0.5), 0.5);
    }

    #[test]
    fn source_emission_uses_the_full_and_candidate_blend() {
        assert_eq!(
            blend_emission(DVec3::splat(10.0), DVec3::splat(2.0), 0.25),
            DVec3::splat(8.0),
        );
    }

    struct EmittingMedium {
        event_count: usize,
        extinction: DVec3,
        scatter: DVec3,
    }

    impl Medium for EmittingMedium {
        fn interval(
            &mut self,
            _start: f64,
            limit: f64,
            _rng: &mut Pcg32,
            _source_rng_offset: u32,
        ) -> Result<Option<Interval>> {
            Ok(Some(Interval {
                end: limit,
                minorant: 0.0,
                majorant: 1_000_000_000.0,
            }))
        }

        fn coefficients(&mut self, _distance: f64) -> Result<Coefficients> {
            self.event_count += 1;
            Ok(Coefficients {
                extinction: self.extinction,
                scatter: self.scatter,
                emission: DVec3::splat(0.4),
            })
        }
        fn extinction(&mut self, _distance: f64) -> Result<DVec3> {
            Ok(self.extinction)
        }
    }

    #[test]
    fn source_transmittance_zero_range_matches_constant_extinction() -> Result<()> {
        let mut medium = EmittingMedium {
            event_count: 0,
            extinction: DVec3::ONE,
            scatter: DVec3::splat(0.5),
        };
        let mut rng = Pcg32 {
            state: 0,
            increment: 1,
            lcg_rng_pixel: 0,
            sample_index: 0,
            source_index: Some(0),
        };
        let result = transmittance(&mut medium, 0.0, 1.0, 0.0, &mut rng, 0, 0)?;
        assert_eq!(result, DVec3::splat((-1.0_f64).exp()));
        assert_eq!(medium.event_count, 0);
        Ok(())
    }

    struct ZeroMajorantMedium {
        coefficient_calls: usize,
        extinction_calls: usize,
    }

    impl Medium for ZeroMajorantMedium {
        fn interval(
            &mut self,
            _start: f64,
            limit: f64,
            _rng: &mut Pcg32,
            _source_rng_offset: u32,
        ) -> Result<Option<Interval>> {
            Ok(Some(Interval {
                end: limit,
                minorant: 0.0,
                majorant: 0.0,
            }))
        }

        fn coefficients(&mut self, _distance: f64) -> Result<Coefficients> {
            self.coefficient_calls += 1;
            Ok(Coefficients {
                extinction: DVec3::ONE,
                scatter: DVec3::splat(0.5),
                emission: DVec3::ZERO,
            })
        }

        fn extinction(&mut self, _distance: f64) -> Result<DVec3> {
            self.extinction_calls += 1;
            Ok(DVec3::ONE)
        }
    }

    #[test]
    fn zero_majorant_skips_mis_eq_but_non_mis_eq_uses_source_transmittance() -> Result<()> {
        let mut medium = ZeroMajorantMedium {
            coefficient_calls: 0,
            extinction_calls: 0,
        };
        let mut mis_samples = [Some(DirectSample {
            distance: 0.5,
            pdf: 0.25,
            payload: String::from("light"),
            throughput: DVec3::ONE,
            distance_pdf: 1.0,
            processed: false,
        })];
        process_direct_samples(
            &mut medium,
            &mut mis_samples,
            0.5,
            0.0,
            0.0,
            true,
            DVec3::ONE,
            1.0,
            1.0,
        )?;
        let mis_sample = mis_samples[0]
            .as_ref()
            .ok_or_else(|| PotError::invalid_operation("missing MIS direct sample"))?;
        assert_eq!(mis_sample.throughput, DVec3::ZERO);
        assert_eq!(mis_sample.distance_pdf, 0.0);
        assert!(mis_sample.processed);
        assert_eq!(mis_sample.pdf, 0.25);
        assert_eq!(mis_sample.payload, "light");
        assert_eq!(medium.coefficient_calls, 0);

        let mut rng = Pcg32::for_sample(1, 2, 3, 4);
        let mut throughput = DVec3::ONE;
        let mut direct_samples = [Some(DirectSample {
            distance: 0.5,
            pdf: 0.25,
            payload: String::from("light"),
            throughput: DVec3::ONE,
            distance_pdf: 1.0,
            processed: false,
        })];
        let segment = trace(
            &mut medium,
            Settings {
                source_rng_offset: 16,
                source_index: 0,
                limit: 1.0,
                events: false,
                use_mis: false,
                guiding: None,
            },
            &mut direct_samples,
            &mut throughput,
            &mut rng,
        )?;

        assert!(matches!(segment.outcome, MediumOutcome::Survive));
        let direct_sample = direct_samples[0]
            .as_ref()
            .ok_or_else(|| PotError::invalid_operation("missing non-MIS direct sample"))?;
        assert_eq!(direct_sample.throughput, DVec3::splat((-0.5_f64).exp()));
        assert_eq!(direct_sample.distance_pdf, 0.0);
        assert!(direct_sample.processed);
        assert_eq!(medium.extinction_calls, 1);
        assert_eq!(medium.coefficient_calls, 0);
        Ok(())
    }

    #[test]
    fn unguided_emission_stops_at_candidate_and_source_allows_1025_events() -> Result<()> {
        let extinction = DVec3::splat(1_000_000_000.0 - 10.0);
        let mut medium = EmittingMedium {
            event_count: 0,
            extinction,
            scatter: extinction,
        };
        let mut rng = Pcg32::for_sample(1, 2, 3, 0);
        let mut throughput = DVec3::ONE;
        let mut direct_samples: [Option<DirectSample<()>>; 0] = [];
        let segment = trace(
            &mut medium,
            Settings {
                limit: 1.0,
                events: true,
                source_rng_offset: 16,
                source_index: 0,
                use_mis: false,
                guiding: None,
            },
            &mut direct_samples,
            &mut throughput,
            &mut rng,
        )?;

        assert!(matches!(segment.outcome, MediumOutcome::Scatter(_)));
        assert!((segment.emission.x - 0.4e-9).abs() < 1.0e-18);
        assert_eq!(medium.event_count, 1025);
        Ok(())
    }

    #[test]
    fn guided_reservoir_uses_source_mixture_and_candidate_weight() -> Result<()> {
        let mut reservoir = Reservoir::new(0.5);
        reservoir.add_sample(
            0.4,
            ReservoirCandidate {
                scatter: ScatterCandidate {
                    distance: 2.0,
                    throughput: DVec3::splat(10.0),
                    distance_pdf: 0.8,
                },
                emission: DVec3::splat(3.0),
            },
        );
        let (candidate, scale, weight, scatter_emission) =
            finalize_guided_reservoir(&mut reservoir, DVec3::splat(2.0), DVec3::splat(4.0), 0.8)
                .ok_or_else(|| PotError::invalid_operation("missing scatter candidate"))?;
        assert!((weight - 0.4).abs() < 1.0e-12);
        assert!((scale - 0.4 / 0.7).abs() < 1.0e-12);
        assert!((candidate.scatter.throughput.x * scale - 40.0 / 7.0).abs() < 1.0e-12);
        assert_eq!(scatter_emission, DVec3::splat(3.0));
        Ok(())
    }

    #[test]
    fn guided_emission_keeps_scatter_candidate_when_transmit_is_selected() -> Result<()> {
        let mut reservoir = Reservoir::new(0.0);
        reservoir.add_sample(
            0.4,
            ReservoirCandidate {
                scatter: ScatterCandidate {
                    distance: 2.0,
                    throughput: DVec3::ONE,
                    distance_pdf: 0.8,
                },
                emission: DVec3::splat(3.0),
            },
        );
        let (candidate, _, weight, scatter_emission) =
            finalize_guided_reservoir(&mut reservoir, DVec3::splat(2.0), DVec3::splat(4.0), 0.8)
                .ok_or_else(|| PotError::invalid_operation("missing finalized candidate"))?;

        assert_eq!(candidate.scatter.distance_pdf, 0.0);
        assert_eq!(candidate.emission, DVec3::splat(4.0));
        assert_eq!(
            blend_emission(candidate.emission, scatter_emission, weight),
            DVec3::splat(3.6),
        );
        Ok(())
    }

    #[test]
    fn zero_transmitted_throughput_forces_scatter_guiding() -> Result<()> {
        let mut reservoir = Reservoir::new(0.5);
        reservoir.add_sample(
            0.2,
            ReservoirCandidate {
                scatter: ScatterCandidate {
                    distance: 1.0,
                    throughput: DVec3::splat(3.0),
                    distance_pdf: 0.4,
                },
                emission: DVec3::ZERO,
            },
        );
        let (candidate, scale, weight, _) =
            finalize_guided_reservoir(&mut reservoir, DVec3::ZERO, DVec3::ZERO, 0.1)
                .ok_or_else(|| PotError::invalid_operation("missing scatter candidate"))?;

        assert_eq!(weight, 0.2);
        assert_eq!(scale, 0.2);
        assert_eq!(candidate.scatter.distance, 1.0);
        assert!(candidate.scatter.distance_pdf > 0.0);
        assert_eq!(reservoir.total_weight, 0.2);
        Ok(())
    }

    struct PiecewiseVacuum;

    impl Medium for PiecewiseVacuum {
        fn interval(
            &mut self,
            start: f64,
            limit: f64,
            _rng: &mut Pcg32,
            _source_rng_offset: u32,
        ) -> Result<Option<Interval>> {
            Ok(Some(if start < 1.0 {
                Interval {
                    end: 1.0,
                    minorant: 0.0,
                    majorant: 2.0,
                }
            } else {
                Interval {
                    end: limit,
                    minorant: 0.0,
                    majorant: 1.0,
                }
            }))
        }

        fn coefficients(&mut self, _distance: f64) -> Result<Coefficients> {
            Ok(Coefficients {
                extinction: DVec3::ZERO,
                scatter: DVec3::ZERO,
                emission: DVec3::ZERO,
            })
        }
        fn extinction(&mut self, _distance: f64) -> Result<DVec3> {
            Ok(DVec3::ZERO)
        }
    }

    #[test]
    fn terminal_walk_sums_unscaled_majorant_once_across_nonzero_intervals() -> Result<()> {
        let mut medium = PiecewiseVacuum;
        let mut rng = Pcg32::for_sample(1, 2, 3, 4);
        let mut throughput = DVec3::splat(2.0);
        let mut direct_samples: [Option<DirectSample<()>>; 0] = [];
        let segment = trace(
            &mut medium,
            Settings {
                limit: 3.0,
                events: false,
                source_rng_offset: 16,
                source_index: 0,
                use_mis: false,
                guiding: Some(GuidingParameters {
                    scatter_probability: 0.7,
                    majorant_scale: 5.0,
                }),
            },
            &mut direct_samples,
            &mut throughput,
            &mut rng,
        )?;

        assert!(matches!(segment.outcome, MediumOutcome::Survive));
        assert_eq!(segment.optical_depth, 4.0);
        assert_eq!(throughput, DVec3::splat(2.0));
        Ok(())
    }

    struct CorrelatedTelescopeMedium {
        interval_offsets: Vec<u32>,
        majorant: f64,
    }

    impl Medium for CorrelatedTelescopeMedium {
        fn interval(
            &mut self,
            start: f64,
            limit: f64,
            _rng: &mut Pcg32,
            source_rng_offset: u32,
        ) -> Result<Option<Interval>> {
            self.interval_offsets.push(source_rng_offset);
            Ok(Some(Interval {
                end: (start.floor() + 1.0).min(limit),
                minorant: 0.0,
                majorant: self.majorant,
            }))
        }

        fn coefficients(&mut self, _distance: f64) -> Result<Coefficients> {
            Ok(Coefficients {
                extinction: DVec3::ZERO,
                scatter: DVec3::ZERO,
                emission: DVec3::ZERO,
            })
        }

        fn extinction(&mut self, distance: f64) -> Result<DVec3> {
            let local_distance = distance - distance.floor();
            let extinction = match (distance < 1.0, local_distance < 0.3) {
                (true, true) => 0.05,
                (true, false) => 0.1,
                (false, true) => 0.15,
                (false, false) => 0.2,
            };
            Ok(DVec3::splat(extinction))
        }
    }

    #[test]
    fn source_expansion_pairs_nonconstant_pixel_estimators() -> Result<()> {
        let mut first_medium = CorrelatedTelescopeMedium {
            interval_offsets: Vec::new(),
            majorant: 0.2,
        };
        let mut second_medium = CorrelatedTelescopeMedium {
            interval_offsets: Vec::new(),
            majorant: 0.2,
        };
        let mut first_samples = TelescopeSamples {
            expansion_sample: None,
            shade_offset: Some(0.25),
        };
        let mut second_samples = TelescopeSamples {
            expansion_sample: None,
            shade_offset: Some(0.75),
        };
        let mut first_rng = Pcg32::for_sample(3, 2, 1, 0);
        let mut second_rng = Pcg32::for_sample(9, 7, 4, 0);

        // Independent Source golden U(sample 0, dimension 6) is 0.0815791935 (order one).
        let first = transmittance_with_samples(
            &mut first_medium,
            0.0,
            1.0,
            1.0,
            0,
            0,
            &mut first_samples,
            &mut first_rng,
        )?;
        let second = transmittance_with_samples(
            &mut second_medium,
            0.0,
            1.0,
            1.0,
            0,
            0,
            &mut second_samples,
            &mut second_rng,
        )?;

        assert!((first.x - 0.924_811_921_937_404_5).abs() < 1.0e-12);
        assert!((second.x - 0.904_837_418_035_959_5).abs() < 1.0e-12);
        assert!((f64::midpoint(first.x, second.x) - 0.914_824_669_986_682).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn source_telescope_shade_offset_uses_pixel_index() -> Result<()> {
        let source_index = 9;
        let source_rng_offset = u32::MAX - 6;
        let mut medium = CorrelatedTelescopeMedium {
            interval_offsets: Vec::new(),
            majorant: 0.0,
        };
        let mut rng = Pcg32::for_sample(1, 2, 3, 0);
        let expected_shade = f64::from(volume_uniform_1d(
            source_index,
            source_rng_offset.wrapping_add(7),
        ));
        let expected_extinction = medium.extinction(expected_shade)?.x;
        let result = transmittance_with_samples(
            &mut medium,
            0.0,
            1.0,
            0.0,
            source_rng_offset,
            source_index,
            &mut TelescopeSamples::default(),
            &mut rng,
        )?;

        assert_eq!(result, DVec3::splat((-expected_extinction).exp()));
        Ok(())
    }

    #[test]
    fn source_transmittance_offset_wraps_before_expansion_dimension() -> Result<()> {
        let mut zero_offset_medium = CorrelatedTelescopeMedium {
            interval_offsets: Vec::new(),
            majorant: 0.2,
        };
        let mut wrapped_offset_medium = CorrelatedTelescopeMedium {
            interval_offsets: Vec::new(),
            majorant: 0.2,
        };
        let mut zero_offset_samples = TelescopeSamples {
            expansion_sample: None,
            shade_offset: Some(0.25),
        };
        let mut wrapped_offset_samples = TelescopeSamples {
            expansion_sample: None,
            shade_offset: Some(0.25),
        };
        let mut zero_offset_rng = Pcg32::for_sample(1, 0, 0, 0);
        let mut wrapped_offset_rng = Pcg32::for_sample(1, 0, 0, 0);

        // Sample 0 gives order one at dimension 6, but order zero at dimension 0 after wrap.
        let zero_offset = transmittance_with_samples(
            &mut zero_offset_medium,
            0.0,
            1.0,
            1.0,
            0,
            0,
            &mut zero_offset_samples,
            &mut zero_offset_rng,
        )?;
        let wrapped_offset = transmittance_with_samples(
            &mut wrapped_offset_medium,
            0.0,
            1.0,
            1.0,
            u32::MAX - 5,
            0,
            &mut wrapped_offset_samples,
            &mut wrapped_offset_rng,
        )?;
        assert!((zero_offset.x - 0.924_811_921_937_404_5).abs() < 1.0e-12);
        assert!((wrapped_offset.x - 0.951_229_424_500_714).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn source_telescope_estimates_are_correlated_across_free_leaves() -> Result<()> {
        let mut medium = CorrelatedTelescopeMedium {
            interval_offsets: Vec::new(),
            majorant: 0.2,
        };
        let mut telescope_samples = TelescopeSamples {
            expansion_sample: None,
            shade_offset: Some(0.25),
        };
        let mut rng = Pcg32::for_sample(5, 2, 3, 0);
        let first = transmittance_with_samples(
            &mut medium,
            0.0,
            1.0,
            0.2,
            0,
            0,
            &mut telescope_samples,
            &mut rng,
        )?;
        let second = transmittance_with_samples(
            &mut medium,
            1.0,
            2.0,
            0.2,
            0,
            0,
            &mut telescope_samples,
            &mut rng,
        )?;

        // Same Source sample/offset and explicit shade point straddle each leaf's extinction step.
        assert!((first.x - 0.924_811_921_937_404_5).abs() < 1.0e-12);
        assert!((second.x - 0.836_804_431_614_715_4).abs() < 1.0e-12);
        assert!((first.x * second.x - 0.773_886_714_687_342_3).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn trace_uses_main_then_walk_offsets_across_free_leaves() -> Result<()> {
        let initial_source_offset = 0x6f3d_3dbc;
        let mut medium = CorrelatedTelescopeMedium {
            interval_offsets: Vec::new(),
            majorant: 0.0,
        };
        let mut rng = Pcg32::for_sample(1, 2, 3, 0);
        let mut throughput = DVec3::ONE;
        let mut direct_samples: [Option<DirectSample<()>>; 0] = [];
        let segment = trace(
            &mut medium,
            Settings {
                limit: 2.0,
                events: false,
                use_mis: false,
                source_rng_offset: initial_source_offset,
                source_index: 0,
                guiding: None,
            },
            &mut direct_samples,
            &mut throughput,
            &mut rng,
        )?;

        assert!(matches!(segment.outcome, MediumOutcome::Survive));
        assert_eq!(medium.interval_offsets, [initial_source_offset, 0]);
        Ok(())
    }

    #[test]
    fn source_interval_offset_advances_after_an_event() -> Result<()> {
        let initial_source_offset = 0x6f3d_3dbc;
        let mut medium = CorrelatedTelescopeMedium {
            interval_offsets: Vec::new(),
            majorant: 32.0,
        };
        let mut rng = Pcg32::for_sample(1, 2, 3, 1);
        let mut throughput = DVec3::ONE;
        let mut direct_samples = [Some(DirectSample {
            distance: 4.0,
            pdf: 1.0,
            payload: (),
            throughput: DVec3::ONE,
            distance_pdf: 0.0,
            processed: false,
        })];
        let segment = trace(
            &mut medium,
            Settings {
                limit: 4.0,
                events: true,
                use_mis: false,
                source_rng_offset: initial_source_offset,
                source_index: 1,
                guiding: None,
            },
            &mut direct_samples,
            &mut throughput,
            &mut rng,
        )?;

        assert!(matches!(segment.outcome, MediumOutcome::Survive));
        let walk_offset = volume_rng_scramble(initial_source_offset, 0xe35f_ad82);
        assert_eq!(medium.interval_offsets[0], initial_source_offset);
        assert!(
            medium.interval_offsets[1..]
                .iter()
                .any(|offset| *offset != walk_offset)
        );
        assert!(medium.interval_offsets[1..].iter().all(|offset| {
            offset.wrapping_sub(walk_offset) % super::super::CYCLES_PRNG_BOUNCE_NUM == 0
        }));
        let direct_sample = direct_samples[0]
            .as_ref()
            .ok_or_else(|| PotError::invalid_operation("missing direct sample"))?;
        assert!(direct_sample.processed);
        Ok(())
    }
}
