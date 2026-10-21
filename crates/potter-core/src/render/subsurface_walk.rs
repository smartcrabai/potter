//! Cycles' random walk subsurface scattering (`kernel/integrator/subsurface_random_walk.h`, with the entry of
//! `subsurface_entry_bounce` in `kernel/integrator/subsurface.h`). The entry refracts the view direction through a GGX
//! interface into the object. A walk of free flights through the interior samples the scattering with Dwivedi's
//! guided phase sampling and the opposite-interface guide, mixing the classic, guided and backward strategies by
//! the balance heuristic over the three colour channels. The walk ends at the object's surface, which is shaded as the
//! diffuse exit of `subsurface.rs`. The distances use d'Eon's van de Hulst mapping from albedo (`subsurface_random_walk_remap`).

use std::f64::consts::PI;

use glam::DVec3;

use super::bvh::{Bvh, BvhHit};
use super::principled_bsdf::{make_orthonormals, sample_visible_normal};
use super::raster::Triangle;
use super::subsurface::SubsurfaceExit;

/// Cycles' `BSSRDF_MAX_BOUNCES` (`kernel/types.h`): the walk ends after this many free flights.
const MAX_BOUNCES: u32 = 256;
/// Bounces up to this index take the object's anisotropy; later ones use the isotropic similarity medium
/// (`SUBSURFACE_RANDOM_WALK_SIMILARITY_LEVEL`, `subsurface_random_walk.h`).
const SIMILARITY_LEVEL: u32 = 9;
/// Cycles' `VOLUME_THROUGHPUT_EPSILON` (`kernel/closure/volume.h`): a walk whose throughput falls below it ends.
const THROUGHPUT_EPSILON: f64 = 1.0e-6;
/// The albedo below which a channel's alpha is raised to this value, with the throughput compensating
/// (`subsurface_random_walk_coefficients`, `min_alpha`).
const MIN_ALPHA: f64 = 0.2;
/// Cycles' floor on a radius before its reciprocal, the extinction (`max(radius, 1e-16f)`, `subsurface_random_walk_remap`).
const RADIUS_FLOOR: f64 = 1.0e-16;

/// The walk's start: the entry point and its normals, the refracted direction, the closure's albedo, radius and
/// anisotropy, and the path throughput, which already includes the entry weight.
pub(super) struct WalkEntry {
    /// The entry point on the surface, `P` in Cycles.
    pub(super) point: DVec3,
    /// The first origin of the walk, a little inside the surface, so the entry triangle is not hit again.
    pub(super) origin: DVec3,
    /// The shading normal, `N` in Cycles, which guides the walk.
    pub(super) normal: DVec3,
    /// The refracted direction, which points into the object.
    pub(super) direction: DVec3,
    pub(super) albedo: DVec3,
    pub(super) radius: DVec3,
    pub(super) anisotropy: f64,
    pub(super) throughput: DVec3,
    /// The distance a ray starts past a hit, as the path tracer's ray epsilon.
    pub(super) epsilon: f64,
}

/// The walk's medium: the extinction and single-scattering albedo per channel, and the throughput factor that divides
/// out the albedo the scattering adds back (`subsurface_random_walk_coefficients`).
struct Medium {
    sigma_t: DVec3,
    alpha: DVec3,
    throughput: DVec3,
}

/// Cycles' `subsurface_random_walk_remap` in colour form (`subsurface_random_walk.h`): d'Eon's Eq. (53.7) from the
/// diffuse colour to the single-scattering albedo, for the anisotropy `g`. The result is clamped below one.
fn albedo_from_colour(color: f64, g: f64) -> f64 {
    let s_sq = (4.20863 * color - (9.59217 + 41.6808 * color + 17.7126 * color * color).sqrt()
        + 4.09712)
        .powi(2);
    let denominator = 1.0 - g * s_sq;
    let alpha = (1.0 - s_sq) / denominator;
    if alpha.is_finite() {
        alpha.clamp(0.0, 0.999_999)
    } else {
        0.0
    }
}

/// Cycles' `subsurface_random_walk_coefficients` with the van de Hulst mapping: the extinction is the reciprocal of
/// the radius, the albedo comes from the colour, and a channel whose alpha is below [`MIN_ALPHA`] is raised to it with
/// its throughput scaled down. The throughput factor divides by the albedo (`safe_divide_color`, zero where it is zero).
fn medium(albedo: DVec3, radius: DVec3, g: f64) -> Medium {
    let mut alpha = DVec3::new(
        albedo_from_colour(albedo.x, g),
        albedo_from_colour(albedo.y, g),
        albedo_from_colour(albedo.z, g),
    );
    let sigma_t = DVec3::new(
        1.0 / radius.x.max(RADIUS_FLOOR),
        1.0 / radius.y.max(RADIUS_FLOOR),
        1.0 / radius.z.max(RADIUS_FLOOR),
    );
    let inverse_albedo = |value: f64| if value > 0.0 { 1.0 / value } else { 0.0 };
    let mut throughput = DVec3::new(
        inverse_albedo(albedo.x),
        inverse_albedo(albedo.y),
        inverse_albedo(albedo.z),
    );
    for channel in 0..3 {
        if alpha[channel] < MIN_ALPHA {
            throughput[channel] *= alpha[channel] / MIN_ALPHA;
            alpha[channel] = MIN_ALPHA;
        }
    }
    Medium {
        sigma_t,
        alpha,
        throughput,
    }
}

/// Dwivedi's diffusion length (Eq. 67 of d'Eon and Křivánek, `diffusion_length_dwivedi`).
fn diffusion_length(alpha: f64) -> f64 {
    1.0 / (1.0 - alpha.powf(2.442_94 - 0.021_581_3 * alpha + 0.578_637 / alpha)).sqrt()
}

/// Cycles' `eval_phase_dwivedi` (Eq. 9 of Meng et al.), with `phase_log = ln((v + 1) / (v - 1))`.
fn eval_phase_dwivedi(v: f64, phase_log: f64, cos: f64) -> f64 {
    1.0 / ((v - cos) * phase_log)
}

/// Cycles' `sample_phase_dwivedi` (Eq. 10 of Meng et al.): the cosine for the uniform `u`.
fn sample_phase_dwivedi(v: f64, phase_log: f64, u: f64) -> f64 {
    v - (v + 1.0) * (-u * phase_log).exp()
}

/// Cycles' `phase_henyey_greenstein` for the cosine between the old and new directions.
fn phase_hg(cos: f64, g: f64) -> f64 {
    if g.abs() < 1e-3 {
        return 1.0 / (4.0 * PI);
    }
    let fac = 1.0 + g * (g - 2.0 * cos);
    (1.0 - g * g) / (4.0 * PI * fac * fac.max(0.0).sqrt())
}

/// A direction at angle `cos` from `direction`, at azimuth `2π·v` (`phase_sample_direction`, `direction_from_cosine`).
fn direction_from_cosine(direction: DVec3, cos: f64, v: f64) -> DVec3 {
    let phi = 2.0 * PI * v;
    let sin = (1.0 - cos * cos).max(0.0).sqrt();
    let (tangent, bitangent) = make_orthonormals(direction);
    tangent * (sin * phi.cos()) + bitangent * (sin * phi.sin()) + direction * cos
}

/// Cycles' `phase_henyey_greenstein_sample`: the direction and its density for the uniform pair `(u, v)`.
fn sample_hg(direction: DVec3, g: f64, u: f64, v: f64) -> (DVec3, f64) {
    let mut cos = 1.0 - 2.0 * u;
    if g.abs() >= 1e-3 {
        let k = (1.0 - g * g) / (1.0 - g * cos);
        cos = (1.0 + g * g - k * k) / (2.0 * g);
    }
    (direction_from_cosine(direction, cos, v), phase_hg(cos, g))
}

/// Cycles' `volume_sample_channel` (`kernel/closure/volume.h`): a channel with probability proportional to
/// `|throughput · albedo|`, and the per-channel probabilities. With no weight, every channel is equally likely.
fn sample_channel(albedo: DVec3, throughput: DVec3, u: f64) -> (usize, DVec3) {
    let weights = (throughput * albedo).abs();
    let sum = weights.element_sum();
    let pdf = if sum > 0.0 {
        weights / sum
    } else {
        DVec3::splat(1.0 / 3.0)
    };
    let mut accumulated = 0.0;
    for channel in 0..2 {
        accumulated += pdf[channel];
        if u < accumulated {
            return (channel, pdf);
        }
    }
    (2, pdf)
}

/// Cycles' `subsurface_random_walk_pdf`: the density of a free flight of length `t` that ends in a hit (the
/// transmittance) or continues (`sigma_t · transmittance`), with the transmittance itself.
fn flight_pdf(sigma_t: DVec3, t: f64, hit: bool) -> (DVec3, DVec3) {
    let transmittance = (-sigma_t * t).exp();
    let pdf = if hit {
        transmittance
    } else {
        sigma_t * transmittance
    };
    (pdf, transmittance)
}

fn mix(first: DVec3, second: DVec3, factor: f64) -> DVec3 {
    first * (1.0 - factor) + second * factor
}

/// The refracted direction of the entry (`subsurface_entry_bounce`): the view direction `view` refracted through a GGX
/// interface of roughness `alpha` onto the object's index `ior` (clamped to `1.01..3.8`, `bssrdf_setup`). `None` when
/// the view is behind the shading normal.
pub(super) fn entry_direction(
    normal: DVec3,
    view: DVec3,
    alpha: f64,
    ior: f64,
    u: f64,
    v: f64,
) -> Option<DVec3> {
    let cos_ni = normal.dot(view);
    if cos_ni <= 0.0 {
        return None;
    }
    let (x, y) = make_orthonormals(normal);
    let local_i = DVec3::new(x.dot(view), y.dot(view), cos_ni);
    let local_h = sample_visible_normal(local_i, alpha, u, v);
    let half = x * local_h.x + y * local_h.y + normal * local_h.z;
    let cos_hi = half.dot(view);
    let neta = 1.0 / ior.clamp(1.01, 3.8);
    let arg = 1.0 - neta * neta * (1.0 - cos_hi * cos_hi);
    let dnp = arg.sqrt().max(1e-7);
    let nk = neta * cos_hi - dnp;
    Some(-(view * neta) + half * nk)
}

/// The random walk from the entry (`subsurface_random_walk`, Cycles). Returns the exit hit on the object, with the
/// origin and direction of the flight that reached it and the path throughput after the walk, or `None` when the walk
/// is absorbed (its throughput falls below [`THROUGHPUT_EPSILON`], it runs out of bounces, or the medium's diffusion
/// length is one). `draw` supplies the uniform numbers of the walk in order.
pub(super) fn walk(
    triangles: &[Triangle],
    bvh: &Bvh,
    object: u32,
    entry: &WalkEntry,
    draw: &mut dyn FnMut() -> f64,
) -> Option<SubsurfaceExit> {
    let g = entry.anisotropy.clamp(-0.99, 0.99);
    let Medium {
        sigma_t: sigma_t_org,
        alpha,
        throughput: albedo_factor,
    } = medium(entry.albedo, entry.radius, g);
    let sigma_s_org = sigma_t_org * alpha;
    let mut throughput = entry.throughput * albedo_factor;

    // The mean free path of the medium is the diffusion length; the guided sampling needs it above one.
    let diffusion = diffusion_length(alpha.max_element());
    if diffusion.is_nan() || diffusion <= 1.0 {
        return None;
    }
    let phase_log = ((diffusion + 1.0) / (diffusion - 1.0)).ln();
    let guided_fraction_org = 1.0 - 0.5_f64.max(g.abs().powf(0.125));
    // The similarity medium of the walk's later bounces (`sigma_s_star`, `sigma_t_star`, `subsurface_random_walk.h`).
    let sigma_s_star = sigma_s_org * (1.0 - g);
    let sigma_t_star = sigma_t_org - sigma_s_org + sigma_s_star;

    let mut position = entry.origin;
    let mut direction = entry.direction;
    let normal = entry.normal;
    let mut have_opposite = false;
    let mut opposite_distance = 0.0;

    for bounce in 0..MAX_BOUNCES {
        let (anisotropy, guided_fraction, sigma_t, sigma_s) = if bounce <= SIMILARITY_LEVEL {
            (g, guided_fraction_org, sigma_t_org, sigma_s_org)
        } else {
            (0.0, 0.75, sigma_t_star, sigma_s_star)
        };

        let (channel, channel_pdf) = sample_channel(alpha, throughput, draw());
        let mut sample_sigma_t = sigma_t[channel];
        let rand_t = draw();

        let mut forward_pdf_factor = 0.0;
        let mut forward_stretching = 1.0;
        let mut backward_pdf_factor = 0.0;
        let mut backward_stretching = 1.0;
        let mut backward_fraction = 0.0;
        // The first flight has its direction from the refraction; later ones pick theirs here.
        if bounce > 0 {
            let guided = draw() < guided_fraction;
            let mut guide_backward = false;
            if have_opposite {
                // The walk's distance from the entry plane, against the opposite interface seen by the first flight.
                let x = (position - entry.point)
                    .dot(-normal)
                    .max(0.0)
                    .min(opposite_distance);
                backward_fraction = 1.0 / (1.0 + ((opposite_distance - 2.0 * x) / diffusion).exp());
                guide_backward = draw() < backward_fraction;
            }
            let u = draw();
            let v = draw();
            let (new_direction, hg_pdf, cos) = if guided {
                let mut cos = sample_phase_dwivedi(diffusion, phase_log, u);
                // The backward guide mirrors the Dwivedi lobe along the normal: the sign of the cosine swaps.
                if guide_backward {
                    cos = -cos;
                }
                let new_direction = direction_from_cosine(normal, cos, v);
                let hg_pdf = phase_hg(direction.dot(new_direction), anisotropy);
                (new_direction, hg_pdf, cos)
            } else {
                let (new_direction, hg_pdf) = sample_hg(direction, anisotropy, u, v);
                (new_direction, hg_pdf, new_direction.dot(normal))
            };
            direction = new_direction;
            forward_pdf_factor =
                eval_phase_dwivedi(diffusion, phase_log, cos) / (2.0 * PI) / hg_pdf;
            backward_pdf_factor =
                eval_phase_dwivedi(diffusion, phase_log, -cos) / (2.0 * PI) / hg_pdf;
            forward_stretching = 1.0 - cos / diffusion;
            backward_stretching = 1.0 + cos / diffusion;
            if guided {
                sample_sigma_t *= if guide_backward {
                    backward_stretching
                } else {
                    forward_stretching
                };
            }
        }

        // The flight's distance. The first flight also looks for the opposite interface up to ten mean free paths.
        let flight = -(1.0 - rand_t).ln() / sample_sigma_t;
        let max_distance = if bounce == 0 {
            flight.max(10.0 / sigma_t.min_element())
        } else {
            flight
        };

        // The nearest hit on the object within the flight (`scene_intersect_local` without a reservoir, so the
        // closest hit).
        let origin = position;
        let mut closest: Option<(f64, BvhHit, DVec3)> = None;
        bvh.walk_hits(
            triangles,
            origin,
            direction,
            entry.epsilon,
            max_distance,
            |hit, from| {
                if triangles
                    .get(hit.triangle_index)
                    .is_some_and(|triangle| triangle.object_index == object)
                {
                    // Hits come nearest first, so the first hit on the object is the closest: the walk stops there.
                    let t = (from - origin).dot(direction) + hit.distance;
                    closest = Some((t, hit, from));
                    return false;
                }
                true
            },
        );

        let exit = if bounce == 0 {
            if let Some((t, _, _)) = closest {
                have_opposite = true;
                // The first flight starts at the entry point, so its opposite distance is the flight's normal component.
                opposite_distance = (direction * t).dot(-normal);
            }
            // The first flight hits only when the closest hit comes before the sampled distance.
            closest.filter(|(t, _, _)| *t < flight)
        } else {
            closest
        };

        let (step, hit) = match exit {
            Some((t, hit, from)) => (t, Some((hit, from))),
            None => (flight, None),
        };
        let hit_here = hit.is_some();
        position = origin + direction * step;

        let (mut pdf, transmittance) = flight_pdf(sigma_t, step, hit_here);
        if bounce > 0 {
            let (mut guided_pdf, _) = flight_pdf(forward_stretching * sigma_t, step, hit_here);
            if have_opposite {
                let (backward_pdf, _) = flight_pdf(backward_stretching * sigma_t, step, hit_here);
                guided_pdf = mix(
                    guided_pdf * forward_pdf_factor,
                    backward_pdf * backward_pdf_factor,
                    backward_fraction,
                );
            } else {
                guided_pdf *= forward_pdf_factor;
            }
            pdf = mix(pdf, guided_pdf, guided_fraction);
        }

        // The MIS of the three channels: the throughput divides by the channel-mixed density (`dot(channel_pdf, pdf)`).
        let numerator = if hit_here {
            transmittance
        } else {
            sigma_s * transmittance
        };
        throughput *= numerator / channel_pdf.dot(pdf);

        if let Some((hit, from)) = hit {
            return Some(SubsurfaceExit {
                hit,
                origin: from,
                direction,
                throughput,
            });
        }
        if throughput.max_element() < THROUGHPUT_EPSILON {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simpson's rule over `[a, b]`.
    fn simpson(a: f64, b: f64, n: usize, f: impl Fn(f64) -> f64) -> f64 {
        let h = (b - a) / n as f64;
        (0..=n)
            .map(|i| {
                let weight = if i == 0 || i == n {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                weight * f(a + h * i as f64)
            })
            .sum::<f64>()
            * h
            / 3.0
    }

    /// A deterministic uniform stream for the tests (a 64-bit xorshift mapped to `[0, 1)`).
    fn uniform_stream(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed | 1;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn albedo_mapping_takes_zero_and_one_to_their_limits() {
        // d'Eon's Eq. (53.7) at colour 0 is 0 and at colour 1 is 1 (`subsurface_random_walk_remap`).
        assert!(albedo_from_colour(0.0, 0.0) < 1e-3);
        assert!((albedo_from_colour(1.0, 0.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn henyey_greenstein_integrates_to_one_over_the_sphere() {
        for g in [0.6, -0.3, 0.0] {
            let total = 2.0 * PI * simpson(-1.0, 1.0, 2000, |cos| phase_hg(cos, g));
            assert!((total - 1.0).abs() < 1e-6, "g {g}: {total}");
        }
    }

    #[test]
    fn henyey_greenstein_samples_have_the_mean_cosine_g() {
        // The mean of cos(θ) under Henyey-Greenstein is g: a check on the sampled cosines.
        let mut uniform = uniform_stream(0x9e37_79b9);
        let direction = DVec3::Z;
        let g = 0.5;
        let samples = 200_000;
        let mean = (0..samples)
            .map(|_| {
                let (new_direction, _) = sample_hg(direction, g, uniform(), uniform());
                new_direction.dot(direction)
            })
            .sum::<f64>()
            / f64::from(samples);
        assert!((mean - g).abs() < 0.01, "mean cosine {mean}");
    }

    #[test]
    fn dwivedi_sampling_spans_the_cosine_range() {
        let diffusion: f64 = 3.0;
        let phase_log = ((diffusion + 1.0) / (diffusion - 1.0)).ln();
        assert!((sample_phase_dwivedi(diffusion, phase_log, 0.0) + 1.0).abs() < 1e-12);
        assert!((sample_phase_dwivedi(diffusion, phase_log, 1.0) - 1.0).abs() < 1e-12);
        // The Dwivedi density integrates to one over the cosine, the normalisation of Eq. (9).
        let total = simpson(-1.0, 1.0, 2000, |cos| {
            eval_phase_dwivedi(diffusion, phase_log, cos)
        });
        assert!((total - 1.0).abs() < 1e-6, "{total}");
    }

    #[test]
    fn channel_is_chosen_in_proportion_to_its_weight() {
        let (channel, pdf) =
            sample_channel(DVec3::new(1.0, 1.0, 1.0), DVec3::new(1.0, 2.0, 1.0), 0.0);
        assert!((pdf.element_sum() - 1.0).abs() < 1e-12);
        assert!((pdf.y - 0.5).abs() < 1e-12);
        assert_eq!(channel, 0);
        // A draw past the first channel's share falls in the second.
        assert_eq!(
            sample_channel(DVec3::ONE, DVec3::new(1.0, 2.0, 1.0), 0.5).0,
            1
        );
    }

    #[test]
    fn medium_raises_a_low_albedo_to_the_minimum_alpha_with_its_throughput() {
        // A low albedo (0.01) gives an alpha below the floor, so the channel is raised to it with its throughput cut.
        let medium = medium(DVec3::new(0.01, 0.9, 0.5), DVec3::ONE, 0.0);
        assert!((medium.alpha.x - MIN_ALPHA).abs() < 1e-12);
        assert!(medium.throughput.x < 1.0 / 0.01);
        assert!((medium.sigma_t.x - 1.0).abs() < 1e-12);
    }
}
