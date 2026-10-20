//! Cycles 5.2.2's Christensen–Burley subsurface profile (`kernel/closure/bssrdf.h`: `bssrdf_burley_eval`,
//! `bssrdf_burley_pdf`, `bssrdf_burley_root_find`, `bssrdf_burley_sample`, `bssrdf_setup_radius`, `bssrdf_sample`,
//! `bssrdf_pdf`), and the disk probe of `kernel/integrator/subsurface_disk.h`.
//!
//! The profile is a radial density in the mean free path `d`. Cycles normalises it over the truncation radius
//! `16 d`, and its mass there is `BURLEY_TRUNCATE_CDF`.
//!
//! Cycles 5.2.2 does not call `bssrdf_burley_setup` (the fitting term of the older Burley model), so the radius is
//! the scaled Principled radius alone. Only the methods Potter evaluates are here; the random walk methods are
//! reported as unsupported.

use std::f64::consts::PI;

use glam::DVec3;

use crate::shader::{BsdfParams, SubsurfaceMethod};

/// Cycles' `BSSRDF_MIN_RADIUS`: a channel with a smaller radius has no subsurface closure and falls back to diffuse.
pub(crate) const MIN_RADIUS: f64 = 1.0e-8;
/// Cycles' `BURLEY_TRUNCATE`: the profile is zero beyond this many mean free paths.
const TRUNCATE: f64 = 16.0;
/// Cycles' `BURLEY_TRUNCATE_CDF`: the profile's mass within `TRUNCATE` mean free paths.
const TRUNCATE_CDF: f64 = 0.996_379_009_370_832_8;

/// The closure a Burley exit is shaded with (`subsurface_shader_data_setup`): a Lambert lobe of weight one, with no
/// other closure, emission or transparency. The exit keeps the entry's normal and geometry only.
pub(crate) fn exit_params(params: BsdfParams) -> BsdfParams {
    BsdfParams {
        base_color: [1.0; 4],
        metallic: 0.0,
        transmission: 0.0,
        specular_ior_level: 0.0,
        coat_weight: 0.0,
        sheen_weight: 0.0,
        subsurface_weight: 0.0,
        emission_strength: 0.0,
        emission_color: [0.0; 3],
        alpha: 1.0,
        transparent_tint: [1.0; 3],
        reflectance: 1.0,
        diffuse_roughness: 0.0,
        thin_film_thickness: 0.0,
        anisotropic_ior: 0.0,
        thin_wall: false,
        volume_scatter: [0.0; 3],
        volume_absorption: [0.0; 3],
        volume_emission: [0.0; 3],
        ..params
    }
}

/// Whether a Principled BSDF with these parameters has a subsurface closure Potter evaluates: a Subsurface Weight
/// above zero under the Burley method. The random walk methods are reported as unsupported and drop out here.
pub(crate) fn burley_active(params: &BsdfParams) -> bool {
    params.subsurface_weight > 0.0 && params.subsurface_method == SubsurfaceMethod::Burley
}

/// The Burley mean free path of each channel: `subsurface_radius × subsurface_scale × 0.25 / π`, clamped at zero
/// (`bssrdf_setup_radius`, with the Principled's radius and scale from `svm/closure.h`).
pub(crate) fn burley_radius(params: &BsdfParams) -> DVec3 {
    DVec3::from_array(params.subsurface_radius).max(DVec3::ZERO)
        * (params.subsurface_scale.max(0.0) * 0.25 / PI)
}

/// The channels that take part in the profile: those with a radius of at least [`MIN_RADIUS`].
pub(crate) fn channel_mask(radius: DVec3) -> [bool; 3] {
    radius.to_array().map(|value| value >= MIN_RADIUS)
}

/// `bssrdf_burley_eval`: the radial profile at distance `r` for mean free path `d` (zero beyond the truncation).
pub(crate) fn profile(d: f64, r: f64) -> f64 {
    if d <= 0.0 || r >= TRUNCATE * d {
        return 0.0;
    }
    let exp_r_3d = (-r / (3.0 * d)).exp();
    let exp_rd = exp_r_3d * exp_r_3d * exp_r_3d;
    (exp_rd + exp_r_3d) / (4.0 * d)
}

/// `bssrdf_burley_pdf`: the profile normalised over the truncation, the density of a sampled radius.
pub(crate) fn pdf(d: f64, r: f64) -> f64 {
    if r == 0.0 {
        return 0.0;
    }
    profile(d, r) / TRUNCATE_CDF
}

/// `bssrdf_burley_root_find`: the scaled radius `x` with `1 - ¼e^{-x} - ¾e^{-x/3} = xi`, by Newton's method from a
/// fit to the curve (Cycles' twelve-step bound is kept: the iteration stops at its tolerance first).
pub(crate) fn root_find(xi: f64) -> f64 {
    const TOLERANCE: f64 = 1.0e-6;
    const MAX_ITERATIONS: usize = 10;
    let mut r = if xi <= 0.9 {
        (xi * xi * 2.4).exp() - 1.0
    } else {
        15.0
    };
    for _ in 0..MAX_ITERATIONS {
        let exp_r_3 = (-r / 3.0).exp();
        let exp_r = exp_r_3 * exp_r_3 * exp_r_3;
        let f = 1.0 - 0.25 * exp_r - 0.75 * exp_r_3 - xi;
        let f_prime = 0.25 * exp_r + 0.25 * exp_r_3;
        if f.abs() < TOLERANCE || f_prime == 0.0 {
            break;
        }
        r = (r - f / f_prime).max(0.0);
    }
    r
}

/// `bssrdf_burley_sample`: the disk radius `r` and the probe height `h` (with `h² + r² = (16 d)²`) for a uniform `xi`.
pub(crate) fn sample_radius(d: f64, xi: f64) -> (f64, f64) {
    let radius = root_find(xi * TRUNCATE_CDF) * d;
    let truncate = TRUNCATE * d;
    (
        radius,
        (truncate * truncate - radius * radius).max(0.0).sqrt(),
    )
}

/// `bssrdf_sample`: picks a channel with a usable radius uniformly from `xi` in `[0, 1)`, and returns it with the
/// number `xi` remapped to `[0, 1)` for the radius. `None` when no channel has a radius.
pub(crate) fn pick_channel(mask: [bool; 3], xi: f64) -> Option<(usize, f64)> {
    let active = mask.iter().filter(|&&on| on).count();
    if active == 0 {
        return None;
    }
    let scaled = (xi * active as f64).min(active as f64 - f64::EPSILON);
    let index = scaled.floor() as usize;
    let channel = mask
        .iter()
        .enumerate()
        .filter(|(_, on)| **on)
        .map(|(channel, _)| channel)
        .nth(index)?;
    Some((channel, scaled - index as f64))
}

/// `bssrdf_eval`: the profile of each channel at `r` (zero for an inactive channel).
pub(crate) fn eval(radius: DVec3, mask: [bool; 3], r: f64) -> DVec3 {
    let d = radius.to_array();
    DVec3::from_array(std::array::from_fn(|channel| {
        if mask[channel] {
            pdf(d[channel], r)
        } else {
            0.0
        }
    }))
}

/// `bssrdf_pdf`: the density of a sampled radius `r`, the mean over the active channels (each is equally likely).
pub(crate) fn pdf_mean(radius: DVec3, mask: [bool; 3], r: f64) -> f64 {
    let active = mask.iter().filter(|&&on| on).count();
    if active == 0 {
        return 0.0;
    }
    eval(radius, mask, r).element_sum() / active as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simpson's rule over `[a, b]`.
    fn simpson(a: f64, b: f64, n: usize, f: impl Fn(f64) -> f64) -> f64 {
        let h = (b - a) / n as f64;
        let mut total = f(a) + f(b);
        for i in 1..n {
            let x = a + h * i as f64;
            total += if i % 2 == 1 { 4.0 } else { 2.0 } * f(x);
        }
        total * h / 3.0
    }

    #[test]
    fn profile_mass_over_the_truncation_is_the_tabulated_cdf() {
        // The profile is a radial density, so its integral over [0, 16 d] is the truncation CDF.
        for d in [0.05, 0.3, 1.7] {
            let mass = simpson(0.0, TRUNCATE * d, 20_000, |r| profile(d, r));
            assert!((mass - TRUNCATE_CDF).abs() < 1.0e-6, "d {d}: mass {mass}");
        }
    }

    #[test]
    fn pdf_is_the_profile_renormalised_over_the_truncation() {
        let d = 0.3;
        for r in [0.01, 0.2, 0.9] {
            assert!((pdf(d, r) * TRUNCATE_CDF - profile(d, r)).abs() < 1.0e-12);
        }
        assert_eq!(pdf(d, 0.0), 0.0);
        assert_eq!(profile(d, TRUNCATE * d), 0.0);
    }

    #[test]
    fn root_find_inverts_the_radial_cdf() {
        // The cdf of the profile is 1 - ¼e^{-x} - ¾e^{-x/3} in units of d.
        for xi in [0.01, 0.2, 0.5, 0.85, 0.95, 0.99] {
            let x = root_find(xi);
            let cdf = 1.0 - 0.25 * (-x).exp() - 0.75 * (-x / 3.0).exp();
            assert!((cdf - xi).abs() < 1.0e-5, "xi {xi}: cdf {cdf} at x {x}");
        }
    }

    #[test]
    fn sampled_radius_has_the_profile_mean() {
        // Stratified draws of the sampler reproduce the profile's first moment over the truncation.
        let d = 0.25;
        let n = 4_000;
        let sampled: f64 = (0..n)
            .map(|i| sample_radius(d, (f64::from(i) + 0.5) / f64::from(n) * TRUNCATE_CDF).0)
            .sum::<f64>()
            / f64::from(n);
        let moment = simpson(0.0, TRUNCATE * d, 20_000, |r| r * pdf(d, r));
        assert!(
            (sampled - moment).abs() / moment < 1.0e-3,
            "sampled {sampled} vs moment {moment}"
        );
    }

    #[test]
    fn probe_height_closes_the_truncation_sphere() {
        let d = 0.4;
        for xi in [0.1, 0.6, 0.9] {
            let (r, h) = sample_radius(d, xi);
            assert!((r * r + h * h - (TRUNCATE * d).powi(2)).abs() < 1.0e-9);
        }
    }

    #[test]
    fn channels_without_a_radius_drop_out_of_the_profile() {
        let radius = DVec3::new(0.5, 0.0, 0.25);
        let mask = channel_mask(radius);
        assert_eq!(mask, [true, false, true]);
        let channel = pick_channel(mask, 0.99).map(|(channel, _)| channel);
        assert_eq!(channel, Some(2));
        assert_eq!(eval(radius, mask, 0.1).y, 0.0);
        assert!(
            (pdf_mean(radius, mask, 0.1) - (pdf(0.5, 0.1) + pdf(0.25, 0.1)) / 2.0).abs() < 1.0e-12
        );
        assert_eq!(pick_channel([false; 3], 0.5), None);
    }

    fn principled(method: SubsurfaceMethod, weight: f64) -> BsdfParams {
        let material = crate::model::Material {
            base_color: [0.8, 0.5, 0.3, 1.0],
            roughness: 0.5,
            ..crate::model::Material::default()
        };
        BsdfParams {
            subsurface_weight: weight,
            subsurface_method: method,
            ..BsdfParams::from_material(&material, DVec3::Z)
        }
    }

    #[test]
    fn only_burley_with_a_weight_is_evaluated() {
        assert!(burley_active(&principled(SubsurfaceMethod::Burley, 0.5)));
        assert!(!burley_active(&principled(SubsurfaceMethod::Burley, 0.0)));
        // The random walk methods are not evaluated: they must not render as Burley.
        assert!(!burley_active(&principled(
            SubsurfaceMethod::RandomWalk,
            0.5
        )));
        assert!(!burley_active(&principled(
            SubsurfaceMethod::RandomWalkSkin,
            0.5
        )));
        assert!(!burley_active(&principled(
            SubsurfaceMethod::RandomWalkLegacy,
            0.5
        )));
        assert!(!burley_active(&principled(SubsurfaceMethod::Mixed, 0.5)));
    }

    #[test]
    fn burley_radius_is_the_principled_radius_times_scale_over_four_pi() {
        // Blender's default radius (1, 0.2, 0.1) and scale 0.005 (`svm/closure.h`, `bssrdf_setup_radius`).
        let radius = burley_radius(&principled(SubsurfaceMethod::Burley, 1.0));
        let factor = 0.005 * 0.25 / PI;
        assert!((radius.x - factor).abs() < 1.0e-15, "{radius:?}");
        assert!((radius.y - 0.2 * factor).abs() < 1.0e-15, "{radius:?}");
        assert!((radius.z - 0.1 * factor).abs() < 1.0e-15, "{radius:?}");
    }
}
