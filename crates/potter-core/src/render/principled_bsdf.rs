//! Principled BSDF lobe model for the path tracer, ported from Cycles v5.2.2 (Blender 5.2.2 is
//! the oracle). Pure functions: random draws stay in `path.rs`.
//!
//! Cycles sources (`intern/cycles`, tag v5.2.2):
//! - `kernel/svm/closure.h` `svm_node_closure_bsdf`, `CLOSURE_BSDF_PRINCIPLED_ID`: metallic lobe
//!   weight `m` (F82-tint Fresnel), transmission weight `t`, weight `(1-m)(1-t)` for the
//!   dielectric specular lobe and the diffuse lobe, with the specular lobe layered over the diffuse
//!   lobe by `closure_layering_weight`. Specular IOR level sets `f0` and `eta`.
//! - `kernel/closure/bsdf_util.h`: `closure_layering_weight`, `fresnel_dielectric_polarized`,
//!   `fresnel_dielectric_Fss`, `fresnel_f82_Fss`, `fresnel_f82tint_B`, `fresnel_f82`.
//! - `kernel/closure/bsdf_microfacet.h`: `generalized_schlick_fresnel` (exponent < 0 remaps the real
//!   dielectric Fresnel onto F0..F90), `bsdf_microfacet_estimate_albedo` (layering albedo from
//!   `ggx_gen_schlick_ior_s`), `microfacet_ggx_preserve_energy` (`ggx_E`/`ggx_Eavg` multiscatter
//!   gain, `Fms` darkening), `bsdf_microfacet_eval` (height-correlated G2, `lambda`).
//! - `node_shader_bsdf_principled.cc`: the default distribution is `SHD_GLOSSY_MULTI_GGX`, so
//!   energy preservation is enabled for the specular and metallic lobes.
//!
//! Estimator contract: `evaluate` is the continuous BSDF (diffuse + specular lobes, reflectance
//! applied). `continuous_density` is the exact density with which `path.rs` draws continuous
//! directions (lobe mixture), so `evaluate / density` is unbiased for the mixture. The transmission
//! lobe stays a delta, handled by `path.rs`.

use std::f64::consts::PI;

use glam::DVec3;

use super::principled_tables::{ggx_energy, ggx_energy_average, ggx_gen_schlick_ior_s};
use crate::shader::{BsdfParams, dielectric_f0};

/// Roughness floor shared with the sampler; Cycles treats roughness below ~1e-5 as delta.
const MIN_ROUGHNESS: f64 = 0.02;

/// Selection probabilities of the three sampling branches, identical to the thresholds
/// `path.rs` uses to pick a branch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LobeProbabilities {
    pub(crate) diffuse: f64,
    pub(crate) specular: f64,
    pub(crate) transmission: f64,
}

pub(crate) fn lobe_probabilities(params: &BsdfParams) -> LobeProbabilities {
    let metallic = params.metallic.clamp(0.0, 1.0);
    let transmission = params.transmission.clamp(0.0, 1.0);
    let diffuse_weight = ((1.0 - metallic) * (1.0 - transmission)).max(0.0);
    let specular_weight = (metallic
        + dielectric_f0(params.ior, params.specular_ior_level) * (1.0 - metallic))
        .max(1.0e-4);
    let total = diffuse_weight + transmission + specular_weight;
    LobeProbabilities {
        diffuse: diffuse_weight / total,
        specular: specular_weight / total,
        transmission: transmission / total,
    }
}

/// Exact density of the continuous (diffuse + specular) directions drawn by `path.rs`, in solid
/// angle. Diffuse is cosine-distributed; specular is the GGX half-vector distribution D·cosθh
/// mapped to the reflected direction.
pub(crate) fn continuous_density(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    light: DVec3,
) -> f64 {
    let probabilities = lobe_probabilities(params);
    let n_dot_l = normal.dot(light).max(0.0);
    let mut density = probabilities.diffuse * n_dot_l / PI;
    let half = (view + light).normalize_or_zero();
    let n_dot_h = normal.dot(half).max(0.0);
    let v_dot_h = view.dot(half).max(0.0);
    if v_dot_h > 0.0 {
        let alpha_squared = ggx_alpha_squared(params.roughness);
        let pdf_half = ggx_distribution(alpha_squared, n_dot_h) * n_dot_h;
        density += probabilities.specular * pdf_half / (4.0 * v_dot_h).max(1.0e-8);
    }
    density
}

/// Diffuse lobe of the Principled BSDF: Lambert under the dielectric-layering weight, times
/// reflectance. It depends on the view only (the layering albedo is view-dependent), so an area
/// light can be integrated against it analytically.
pub(crate) fn diffuse_coefficient(params: &BsdfParams, normal: DVec3, view: DVec3) -> DVec3 {
    let n_dot_v = normal.dot(view).max(0.0);
    if n_dot_v <= 0.0 {
        return DVec3::ZERO;
    }
    let roughness = params.roughness.clamp(MIN_ROUGHNESS, 1.0);
    let metallic = params.metallic.clamp(0.0, 1.0);
    let transmission = params.transmission.clamp(0.0, 1.0);
    let dielectric_weight = ((1.0 - metallic) * (1.0 - transmission)).max(0.0);
    let diffuse_weight = match specular_layer(params) {
        Some(layer) => {
            let z = ((layer.eta - 1.0) / (layer.eta + 1.0)).abs().sqrt();
            let albedo_remap = ggx_gen_schlick_ior_s(roughness, n_dot_v, z);
            let albedo = layer.f0_tinted.lerp(DVec3::ONE, albedo_remap);
            closure_layering_weight(albedo.max_element(), dielectric_weight)
        }
        None => dielectric_weight,
    };
    base_color(params) * (diffuse_weight / PI) * params.reflectance.clamp(0.0, 1.0)
}

/// Specular (dielectric and metallic) lobes along `light`, times reflectance. Zero when the
/// light or view is below the surface.
pub(crate) fn specular_coefficient(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    light: DVec3,
) -> DVec3 {
    let n_dot_v = normal.dot(view).max(0.0);
    let n_dot_l = normal.dot(light).max(0.0);
    if n_dot_v <= 0.0 || n_dot_l <= 0.0 {
        return DVec3::ZERO;
    }
    let half = (view + light).normalize_or_zero();
    let n_dot_h = normal.dot(half).max(0.0);
    let v_dot_h = view.dot(half).max(0.0);
    let roughness = params.roughness.clamp(MIN_ROUGHNESS, 1.0);
    let alpha_squared = ggx_alpha_squared(roughness);
    let metallic = params.metallic.clamp(0.0, 1.0);
    let transmission = params.transmission.clamp(0.0, 1.0);
    let base = base_color(params);

    // Energy preservation: E is looked up at the view direction, as Cycles does at setup
    // (`microfacet_ggx_preserve_energy(kg, bsdf, sd->wi, ...)`), so it is constant per shading point.
    let energy = ggx_energy(roughness, n_dot_v).max(1.0e-6);
    let energy_average = ggx_energy_average(roughness);
    let missing = (1.0 - energy) / energy;

    let common = ggx_distribution(alpha_squared, n_dot_h)
        * ggx_height_correlated_g2(alpha_squared, n_dot_v, n_dot_l)
        / (4.0 * n_dot_v * n_dot_l).max(1.0e-8);

    let dielectric_weight = ((1.0 - metallic) * (1.0 - transmission)).max(0.0);
    let mut reflected = DVec3::ZERO;

    if let Some(layer) = specular_layer(params) {
        let f0 = layer.f0_tinted;
        let real_f0 = f0_from_ior(layer.eta);
        let real_f = fresnel_dielectric(v_dot_h, layer.eta);
        let remap = saturate(inverse_lerp(real_f0, 1.0, real_f));
        let fresnel = f0.lerp(DVec3::ONE, remap);

        let fss_remap = saturate(inverse_lerp(
            real_f0,
            1.0,
            fresnel_dielectric_fss(layer.eta),
        ));
        let fss = f0.lerp(DVec3::ONE, fss_remap);
        let gain = multiscatter_gain(fss, energy_average, missing);
        reflected += gain * fresnel * dielectric_weight;
    }

    if metallic > 0.0 {
        let tint = DVec3::new(
            params.specular_tint[0],
            params.specular_tint[1],
            params.specular_tint[2],
        )
        .max(DVec3::ZERO)
        .min(DVec3::ONE);
        let b = if tint == DVec3::ONE {
            DVec3::ZERO
        } else {
            f82_tint_b(base, tint)
        };
        let fresnel = f82_fresnel(v_dot_h, base, b);
        let fss = f82_fss(base, b);
        let gain = multiscatter_gain(fss, energy_average, missing);
        reflected += gain * fresnel * metallic;
    }

    (reflected * common) * params.reflectance.clamp(0.0, 1.0)
}

/// Continuous Principled BSDF times reflectance, the `evaluate / density` estimator's numerator.
pub(crate) fn evaluate(params: &BsdfParams, normal: DVec3, view: DVec3, light: DVec3) -> DVec3 {
    if normal.dot(light) <= 0.0 {
        return DVec3::ZERO;
    }
    diffuse_coefficient(params, normal, view) + specular_coefficient(params, normal, view, light)
}

fn base_color(params: &BsdfParams) -> DVec3 {
    DVec3::new(
        params.base_color[0],
        params.base_color[1],
        params.base_color[2],
    )
    .clamp(DVec3::ZERO, DVec3::ONE)
}

/// `closure_layering_weight(layer_albedo, weight)` for a scalar weight and a per-channel albedo:
/// `weight * saturate(1 - max(albedo / weight))`, i.e. `max(0, weight - max(albedo))`.
fn closure_layering_weight(max_layer_albedo: f64, weight: f64) -> f64 {
    if weight <= 0.0 {
        return 0.0;
    }
    weight * (1.0 - max_layer_albedo / weight).clamp(0.0, 1.0)
}

struct SpecularLayer {
    eta: f64,
    f0_tinted: DVec3,
}

/// Cycles' specular setup: `f0 = F0(ior) * 2 * level`, `eta = ior` at level 0.5, otherwise
/// `eta = ior_from_F0(f0)` (inverted for ior < 1). No specular layer when `eta == 1`.
fn specular_layer(params: &BsdfParams) -> Option<SpecularLayer> {
    let level = params.specular_ior_level.max(0.0);
    let f0_scalar = f0_from_ior(params.ior) * 2.0 * level;
    let eta = if level == 0.5 {
        params.ior
    } else {
        let eta = ior_from_f0(f0_scalar);
        if params.ior < 1.0 { 1.0 / eta } else { eta }
    };
    if eta == 1.0 {
        return None;
    }
    let tint = DVec3::new(
        params.specular_tint[0],
        params.specular_tint[1],
        params.specular_tint[2],
    )
    .max(DVec3::ZERO);
    let f0_tinted = (tint * f0_scalar).clamp(DVec3::ZERO, DVec3::ONE);
    Some(SpecularLayer {
        eta,
        f0_tinted,
    })
}

/// Energy-compensation darkening, `1 + Fms * (1 - E) / E` (Cycles `microfacet_ggx_preserve_energy`).
fn multiscatter_gain(fss: DVec3, energy_average: f64, missing: f64) -> DVec3 {
    let fms = fss * energy_average / (DVec3::ONE - fss * (1.0 - energy_average));
    DVec3::ONE + fms * missing
}

fn ggx_alpha_squared(roughness: f64) -> f64 {
    let alpha = roughness.clamp(MIN_ROUGHNESS, 1.0).powi(2);
    alpha * alpha
}

fn ggx_distribution(alpha_squared: f64, n_dot_h: f64) -> f64 {
    let cos_squared = n_dot_h.powi(2).min(1.0);
    let one_minus = 1.0 - cos_squared;
    alpha_squared / (PI * (one_minus + alpha_squared * cos_squared).powi(2)).max(1.0e-12)
}

fn ggx_lambda(alpha_squared: f64, cosine: f64) -> f64 {
    let cosine = cosine.max(1.0e-8);
    0.5 * ((1.0 + alpha_squared * (1.0 / (cosine * cosine) - 1.0).max(0.0)).sqrt() - 1.0)
}

fn ggx_height_correlated_g2(alpha_squared: f64, n_dot_v: f64, n_dot_l: f64) -> f64 {
    1.0 / (1.0 + ggx_lambda(alpha_squared, n_dot_v) + ggx_lambda(alpha_squared, n_dot_l))
}

fn saturate(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

/// Cycles `inverse_lerp(a, b, x) = (x - a) / (b - a)`.
fn inverse_lerp(a: f64, b: f64, x: f64) -> f64 {
    (x - a) / (b - a)
}

pub(crate) fn f0_from_ior(ior: f64) -> f64 {
    ((ior - 1.0) / (ior + 1.0)).powi(2)
}

fn ior_from_f0(f0: f64) -> f64 {
    let root = f0.clamp(0.0, 0.99).sqrt();
    (1.0 + root) / (1.0 - root)
}

/// Unpolarised dielectric Fresnel reflectance (`fresnel_dielectric`, average of s and p).
fn fresnel_dielectric(cosine_incident: f64, eta: f64) -> f64 {
    let cos_i = cosine_incident.abs();
    let eta_cos_t_squared = eta * eta - (1.0 - cos_i * cos_i);
    if eta_cos_t_squared <= 0.0 {
        return 1.0;
    }
    let cos_t = -eta_cos_t_squared.sqrt() / eta;
    let r_s = (cos_i + eta * cos_t) / (cos_i - eta * cos_t);
    let r_p = (cos_t + eta * cos_i) / (eta * cos_i - cos_t);
    0.5 * (r_s * r_s + r_p * r_p)
}

/// Kulla-Conty fit of the average single-scattering dielectric Fresnel (`fresnel_dielectric_Fss`).
fn fresnel_dielectric_fss(eta: f64) -> f64 {
    if eta < 1.0 {
        0.997118 + eta * (0.1014 - eta * (0.965241 + eta * 0.130607))
    } else {
        (eta - 1.0) / (4.08567 + 1.00071 * eta)
    }
}

/// Schlick-with-F82 metallic Fresnel (`fresnel_f82`), evaluated at `cos(H, V)`.
fn f82_fresnel(cosine: f64, f0: DVec3, b: DVec3) -> DVec3 {
    let s = saturate(1.0 - cosine);
    let s5 = s.powi(5);
    let schlick = f0.lerp(DVec3::ONE, s5);
    (schlick - b * cosine * s5 * s).clamp(DVec3::ZERO, DVec3::ONE)
}

/// `fresnel_f82tint_B`: correction term that makes the tint act at 82 degrees.
fn f82_tint_b(f0: DVec3, tint: DVec3) -> DVec3 {
    let f = 6.0 / 7.0;
    let f5 = f.powi(5);
    let schlick = f0.lerp(DVec3::ONE, f5);
    schlick * (7.0 / (f5 * f)) * (DVec3::ONE - tint)
}

/// `fresnel_f82_Fss`: average single-scattering Fresnel of the F82 lobe.
fn f82_fss(f0: DVec3, b: DVec3) -> DVec3 {
    f0.lerp(DVec3::ONE, 1.0 / 21.0) - b * (1.0 / 126.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layering_weight_matches_cycles_formula() {
        // weight * saturate(1 - albedo / weight) for a scalar weight.
        assert!((closure_layering_weight(0.04, 1.0) - 0.96).abs() < 1.0e-12);
        assert!((closure_layering_weight(0.04, 0.76) - 0.72).abs() < 1.0e-12);
        assert_eq!(closure_layering_weight(0.5, 0.0), 0.0);
        assert_eq!(closure_layering_weight(2.0, 1.0), 0.0);
    }

    #[test]
    fn dielectric_fresnel_at_normal_incidence_is_f0() {
        let eta = 1.5;
        assert!((fresnel_dielectric(1.0, eta) - f0_from_ior(eta)).abs() < 1.0e-12);
    }

    #[test]
    fn f82_without_tint_is_schlick_at_normal_incidence() {
        let f0 = DVec3::new(0.2, 0.4, 0.8);
        let b = f82_tint_b(f0, DVec3::ONE);
        assert_eq!(b, DVec3::ZERO);
        assert!((f82_fresnel(1.0, f0, b) - f0).abs().max_element() < 1.0e-12);
    }

    #[test]
    fn multiscatter_gain_is_one_without_missing_energy() {
        let gain = multiscatter_gain(DVec3::splat(0.5), 1.0, 0.0);
        assert_eq!(gain, DVec3::ONE);
    }
}
