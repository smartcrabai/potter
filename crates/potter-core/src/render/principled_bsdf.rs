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

/// Selection probabilities of the four sampling branches, identical to the thresholds `path.rs`
/// uses to pick a branch: transmission, diffuse, coat, specular.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LobeProbabilities {
    pub(crate) diffuse: f64,
    pub(crate) coat: f64,
    pub(crate) specular: f64,
    pub(crate) transmission: f64,
}

pub(crate) fn lobe_probabilities(params: &BsdfParams, view: DVec3) -> LobeProbabilities {
    lobe_probabilities_of(params, coat_layer(params, view).as_ref())
}

/// The sampling branch probabilities for a coat layer that is already set up.
fn lobe_probabilities_of(params: &BsdfParams, coat: Option<&CoatLayer>) -> LobeProbabilities {
    let metallic = params.metallic.clamp(0.0, 1.0);
    let transmission = params.transmission.clamp(0.0, 1.0);
    let diffuse_weight = ((1.0 - metallic) * (1.0 - transmission)).max(0.0);
    // The coat's layering albedo is its selection weight: Cycles' sample weight of the coat closure.
    let coat_weight = coat.map_or(0.0, |coat| coat.albedo);
    let specular_weight = (metallic
        + dielectric_f0(params.ior, params.specular_ior_level) * (1.0 - metallic))
        .max(1.0e-4);
    let total = diffuse_weight + coat_weight + transmission + specular_weight;
    LobeProbabilities {
        diffuse: diffuse_weight / total,
        coat: coat_weight / total,
        specular: specular_weight / total,
        transmission: transmission / total,
    }
}

/// Exact density of the continuous (diffuse, coat and specular) directions drawn by `path.rs`, in
/// solid angle. Diffuse is cosine-distributed; the coat and specular lobes are the GGX half-vector
/// distribution D·cosθh mapped to the reflected direction, with their own roughness. The coat layer
/// is set up once for the whole density.
pub(crate) fn continuous_density(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    light: DVec3,
) -> f64 {
    let coat = coat_layer(params, view);
    let probabilities = lobe_probabilities_of(params, coat.as_ref());
    let mut density = probabilities.diffuse * normal.dot(light).max(0.0) / PI;
    if let Some(coat) = coat {
        density += probabilities.coat * half_vector_pdf(coat.normal, coat.roughness, view, light);
    }
    density + probabilities.specular * half_vector_pdf(normal, params.roughness, view, light)
}

/// Density in solid angle of the reflection of `view` about the half vector drawn with
/// `sample_ggx_half`: `D(h)·(n·h) / (4 v·h)` for the half vector of `view` and `light`.
fn half_vector_pdf(normal: DVec3, roughness: f64, view: DVec3, light: DVec3) -> f64 {
    let half = (view + light).normalize_or_zero();
    let n_dot_h = normal.dot(half).max(0.0);
    let v_dot_h = view.dot(half).max(0.0);
    if v_dot_h <= 0.0 {
        return 0.0;
    }
    let pdf_half = ggx_distribution(ggx_alpha_squared(roughness), n_dot_h) * n_dot_h;
    pdf_half / (4.0 * v_dot_h).max(1.0e-8)
}

/// Diffuse lobe of the Principled BSDF: Lambert under the dielectric-layering weight, times
/// reflectance. It depends on the view only (the layering albedo is view-dependent), so an area
/// light can be integrated against it analytically.
pub(crate) fn diffuse_coefficient(params: &BsdfParams, normal: DVec3, view: DVec3) -> DVec3 {
    diffuse_lobe(params, normal, view, coat_lower_factor(params, view))
}

/// The diffuse lobe, given the coat's lower-layer factor (set up once by the caller).
fn diffuse_lobe(params: &BsdfParams, normal: DVec3, view: DVec3, lower: DVec3) -> DVec3 {
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
            let energy = ggx_energy(roughness, n_dot_v).max(1.0e-6);
            let darkened = albedo * dielectric_darkening(&layer, roughness, energy);
            layered_diffuse_weight(dielectric_weight, darkened.max_element())
        }
        None => dielectric_weight,
    };
    diffuse_albedo(params) * (diffuse_weight / PI) * params.reflectance.clamp(0.0, 1.0) * lower
}

/// Specular (dielectric and metallic) lobes along `light`, times reflectance. Zero when the
/// light or view is below the surface.
pub(crate) fn specular_coefficient(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    light: DVec3,
) -> DVec3 {
    specular_lobe(params, normal, view, light, coat_lower_factor(params, view))
}

/// The specular lobe, given the coat's lower-layer factor (set up once by the caller).
fn specular_lobe(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    light: DVec3,
    lower: DVec3,
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
        let real_f0 = dielectric_f0(layer.eta, 0.5);
        let real_f = fresnel_dielectric(v_dot_h, layer.eta);
        let remap = saturate(inverse_lerp(real_f0, 1.0, real_f));
        let fresnel = f0.lerp(DVec3::ONE, remap);
        let gain = multiscatter_gain(dielectric_fss(&layer), energy_average, missing);
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

    (reflected * common) * (params.reflectance.clamp(0.0, 1.0) * lower)
}

/// Cycles' `CLOSURE_WEIGHT_CUTOFF`: a coat weight at or below it is not set up.
const COAT_WEIGHT_CUTOFF: f64 = 1.0e-5;

/// Continuous Principled BSDF times reflectance, the `evaluate / density` estimator's numerator.
/// The coat layer is set up once here and shared by the lobes it attenuates.
pub(crate) fn evaluate(params: &BsdfParams, normal: DVec3, view: DVec3, light: DVec3) -> DVec3 {
    if normal.dot(light) <= 0.0 {
        return DVec3::ZERO;
    }
    let coat = coat_layer(params, view);
    let lower = coat_lower_factor_of(params, coat.as_ref());
    diffuse_lobe(params, normal, view, lower)
        + specular_lobe(params, normal, view, light, lower)
        + coat_term(params, coat.as_ref(), view, light)
}

/// The Principled coat: a GGX dielectric reflection lobe over the layers below it, set up as
/// Cycles' coat block of `principled_bsdf_emission` (`kernel/svm/closure.h`). `albedo` is the
/// layering albedo `weight · F · gain · E`. `gain` is the multiscatter gain `1 + Fms · (1 - E) / E`
/// of `microfacet_ggx_preserve_energy`: its darkening of the closure weight cancels the
/// `energy_scale` of the eval, so the lobe carries `gain` alone.
#[derive(Clone, Copy, Debug)]
struct CoatLayer {
    normal: DVec3,
    n_dot_v: f64,
    ior: f64,
    roughness: f64,
    weight: f64,
    gain: f64,
    albedo: f64,
}

/// The coat layer for `view`, or `None` when the coat is off or the view is below its normal. The
/// normal is the linked Coat Normal, or the final shading normal when unlinked.
fn coat_layer(params: &BsdfParams, view: DVec3) -> Option<CoatLayer> {
    let weight = params.coat_weight.max(0.0);
    if weight <= COAT_WEIGHT_CUTOFF {
        return None;
    }
    let normal = params.coat_normal_or_shading();
    let n_dot_v = normal.dot(view);
    if n_dot_v <= 0.0 {
        return None;
    }
    let ior = params.coat_ior.max(1.0);
    let roughness = saturate(params.coat_roughness).clamp(MIN_ROUGHNESS, 1.0);
    let energy = ggx_energy(roughness, n_dot_v).max(1.0e-6);
    let missing = (1.0 - energy) / energy;
    let fss = DVec3::splat(fresnel_dielectric_fss(ior));
    let gain = multiscatter_gain(fss, ggx_energy_average(roughness), missing).x;
    // Cycles' DIELECTRIC albedo: the `ggx_gen_schlick_ior_s` LUT for ior > 1, where ior 1 has no
    // Fresnel reflectance at all.
    let fresnel_albedo = if ior > 1.0 {
        let f0 = dielectric_f0(ior, 0.5);
        let z = ((ior - 1.0) / (ior + 1.0)).abs().sqrt();
        let s = ggx_gen_schlick_ior_s(roughness, n_dot_v, z);
        f0 + (1.0 - f0) * s
    } else {
        0.0
    };
    Some(CoatLayer {
        normal,
        n_dot_v,
        ior,
        roughness,
        weight,
        gain,
        albedo: weight * fresnel_albedo * gain * energy,
    })
}

/// The factor the coat leaves to the layers below it: Cycles' `closure_layering_weight` of the coat
/// albedo, times the tint `mix(1, tint^(1/cos θt), coat weight)` (`kernel/svm/closure.h`). The
/// diffuse, specular, metallic and transmission lobes and the emission all scale by it.
pub(crate) fn coat_lower_factor(params: &BsdfParams, view: DVec3) -> DVec3 {
    coat_lower_factor_of(params, coat_layer(params, view).as_ref())
}

/// `coat_lower_factor` for a coat layer that is already set up.
fn coat_lower_factor_of(params: &BsdfParams, coat: Option<&CoatLayer>) -> DVec3 {
    let Some(coat) = coat else {
        return DVec3::ONE;
    };
    let layering = saturate(1.0 - coat.albedo);
    let cos_ni = coat.n_dot_v;
    let cos_nt = (1.0 - (1.0 / coat.ior).powi(2) * (1.0 - cos_ni * cos_ni))
        .max(1.0e-8)
        .sqrt();
    let tint = DVec3::from_array(params.coat_tint).max(DVec3::ZERO);
    let transmitted = tint.powf(1.0 / cos_nt);
    DVec3::ONE.lerp(transmitted, coat.weight) * layering
}

/// The coat lobe along `light`, times reflectance. It is zero when the coat is off or the light is
/// below the coat's normal.
pub(crate) fn coat_coefficient(params: &BsdfParams, view: DVec3, light: DVec3) -> DVec3 {
    coat_term(params, coat_layer(params, view).as_ref(), view, light)
}

/// The coat lobe for a coat layer that is already set up.
fn coat_term(params: &BsdfParams, coat: Option<&CoatLayer>, view: DVec3, light: DVec3) -> DVec3 {
    let Some(coat) = coat else {
        return DVec3::ZERO;
    };
    let normal = coat.normal;
    let n_dot_v = coat.n_dot_v;
    let n_dot_l = normal.dot(light);
    if n_dot_l <= 0.0 {
        return DVec3::ZERO;
    }
    let half = (view + light).normalize_or_zero();
    let n_dot_h = normal.dot(half).max(0.0);
    let v_dot_h = view.dot(half).max(0.0);
    let alpha_squared = ggx_alpha_squared(coat.roughness);
    let common = ggx_distribution(alpha_squared, n_dot_h)
        * ggx_height_correlated_g2(alpha_squared, n_dot_v, n_dot_l)
        / (4.0 * n_dot_v * n_dot_l).max(1.0e-8);
    let fresnel = fresnel_dielectric(v_dot_h, coat.ior);
    DVec3::splat(coat.weight * coat.gain * fresnel * common) * params.reflectance.clamp(0.0, 1.0)
}

/// The base colour clamped to 1 (Cycles' `clamped_base_color`, used by the F82 metal lobe).
fn base_color(params: &BsdfParams) -> DVec3 {
    diffuse_albedo(params).min(DVec3::ONE)
}

/// The diffuse albedo: the base colour clamped at zero only. Cycles' diffuse closure takes the
/// unclamped base colour (`svm_closure.h`, the `bsdf_oren_nayar_setup` call); the clamp to 1 is
/// applied for the F82 metal lobe only.
fn diffuse_albedo(params: &BsdfParams) -> DVec3 {
    DVec3::new(
        params.base_color[0],
        params.base_color[1],
        params.base_color[2],
    )
    .max(DVec3::ZERO)
}

/// `closure_layering_weight` for the diffuse lobe under a dielectric layer. `layer_albedo` is the
/// layer's albedo with its energy darkening applied (the weighted albedo the closure reports), so
/// the diffuse lobe keeps `weight · (1 - albedo)` of its weight.
fn layered_diffuse_weight(weight: f64, layer_albedo: f64) -> f64 {
    weight * saturate(1.0 - layer_albedo)
}

/// Cycles' `microfacet_ggx_preserve_energy` darkening of the dielectric specular closure weight,
/// `(1 + Fms · m) / (1 + m)` with `m = (1 - E) / E`, which equals `gain · E`.
fn dielectric_darkening(layer: &SpecularLayer, roughness: f64, energy: f64) -> DVec3 {
    let missing = (1.0 - energy) / energy;
    multiscatter_gain(
        dielectric_fss(layer),
        ggx_energy_average(roughness),
        missing,
    ) * energy
}

/// The average single-scattering Fresnel of the dielectric layer, remapped from F0..1 as the
/// specular lobe's energy compensation uses it (`fresnel_dielectric_Fss` through
/// `generalized_schlick_fresnel`).
fn dielectric_fss(layer: &SpecularLayer) -> DVec3 {
    let real_f0 = dielectric_f0(layer.eta, 0.5);
    let fss_remap = saturate(inverse_lerp(
        real_f0,
        1.0,
        fresnel_dielectric_fss(layer.eta),
    ));
    layer.f0_tinted.lerp(DVec3::ONE, fss_remap)
}

struct SpecularLayer {
    eta: f64,
    f0_tinted: DVec3,
}

/// Cycles' specular setup: `f0 = F0(ior) * 2 * level`, `eta = ior` at level 0.5, otherwise
/// `eta = ior_from_F0(f0)` (inverted for ior < 1). No specular layer when `eta == 1`.
#[expect(
    clippy::float_cmp,
    reason = "Cycles compares the specular IOR level with exactly 0.5 and eta with exactly 1"
)]
fn specular_layer(params: &BsdfParams) -> Option<SpecularLayer> {
    let level = params.specular_ior_level.max(0.0);
    let f0_scalar = dielectric_f0(params.ior, level);
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
    Some(SpecularLayer { eta, f0_tinted })
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

fn ior_from_f0(f0: f64) -> f64 {
    let root = f0.clamp(0.0, 0.99).sqrt();
    (1.0 + root) / (1.0 - root)
}

/// Unpolarised dielectric Fresnel reflectance (`fresnel_dielectric`, average of s and p).
#[expect(
    clippy::manual_midpoint,
    reason = "Cycles averages the s and p reflectances as 0.5 * (s + p), so the same expression is kept"
)]
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
        0.997_118 + eta * (0.1014 - eta * (0.965_241 + eta * 0.130_607))
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
    let f: f64 = 6.0 / 7.0;
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
    fn layered_diffuse_weight_keeps_what_the_layer_does_not_reflect() {
        // weight · (1 - layer albedo), with the albedo already darkened.
        assert!((layered_diffuse_weight(1.0, 0.04) - 0.96).abs() < 1.0e-12);
        assert!((layered_diffuse_weight(0.76, 0.04) - 0.7296).abs() < 1.0e-12);
        assert_eq!(layered_diffuse_weight(0.0, 0.5), 0.0);
        assert_eq!(layered_diffuse_weight(1.0, 2.0), 0.0);
    }

    #[test]
    fn dielectric_fresnel_at_normal_incidence_is_f0() {
        let eta = 1.5;
        assert!((fresnel_dielectric(1.0, eta) - dielectric_f0(eta, 0.5)).abs() < 1.0e-12);
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

#[cfg(test)]
mod single_bounce_tests {
    use super::*;
    use crate::model::Material;

    // Geometry of tests/fixtures/native_oracle/principled/single_bounce_probe.py: a receiver at the
    // origin with normal +Z under a disk emitter of radius 1.2 and flux 1000 W, radiance
    // P / (π · area) (Cycles' AreaLight convention), 4 units away on the mirror of a view 45°
    // from the normal. The Blender pixel is the single-bounce integral of the lobe.
    const VIEW_THETA: f64 = std::f64::consts::FRAC_PI_4;
    const LIGHT_DISTANCE: f64 = 4.0;
    const RADIUS: f64 = 1.2;
    const POWER: f64 = 1000.0;
    /// Quadrature of the disk in polar coordinates. `COARSE` (96 radial × 192 angular midpoints)
    /// agrees with a converged square grid to about 1e-4 relative on the diffuse, sheen and metal
    /// variants. On the coat it is 9e-5 at roughness 0.24 and 1.7e-3 at roughness 0.1, so the coat
    /// oracles use `FINE` (384 × 768: 5e-7 and 1.0e-4 respectively).
    #[derive(Clone, Copy)]
    struct Steps {
        radial: usize,
        angular: usize,
    }
    const COARSE: Steps = Steps {
        radial: 96,
        angular: 192,
    };
    const FINE: Steps = Steps {
        radial: 384,
        angular: 768,
    };
    /// Orthographic width of the probe camera (Blender's `ortho_scale`) in units.
    const ORTHO_SCALE: f64 = 0.2;

    /// Mean over the central 8×8 pixel patch of the orthographic probe (16 pixels across 0.2 units).
    /// The camera views the plane at `VIEW_THETA`, so the image's vertical axis maps to receiver x
    /// with a factor `cos(VIEW_THETA)`: one pixel is `pixel / cos` in x and `pixel` in y. A glossy
    /// highlight depends on the receiver position, so the patch is averaged, as Blender's pixel is.
    /// The probe is symmetric under y -> -y, so rows j and 7-j contribute equally.
    fn probe_pixel(params: &BsdfParams, steps: Steps) -> DVec3 {
        let pixel = ORTHO_SCALE / 16.0;
        let x_step = pixel / VIEW_THETA.cos();
        let mut total = DVec3::ZERO;
        for i in 0..8 {
            for j in 0..4 {
                let x = (f64::from(i) - 3.5) * x_step;
                let y = (f64::from(j) - 3.5) * pixel;
                total += 2.0 * receiver_integral(params, x, y, steps);
            }
        }
        total / 64.0
    }

    /// Single-bounce integral at the receiver `(x, y, 0)`.
    fn receiver_integral(params: &BsdfParams, x: f64, y: f64, steps: Steps) -> DVec3 {
        let normal = DVec3::Z;
        let view = DVec3::new(VIEW_THETA.sin(), 0.0, VIEW_THETA.cos());
        let centre = LIGHT_DISTANCE * DVec3::new(-VIEW_THETA.sin(), 0.0, VIEW_THETA.cos());
        let toward_receiver = -centre.normalize();
        let u = toward_receiver.cross(DVec3::Y).normalize();
        let w = toward_receiver.cross(u);
        let receiver = DVec3::new(x, y, 0.0);
        let radiance = POWER / (std::f64::consts::PI * std::f64::consts::PI * RADIUS * RADIUS);
        let mut total = DVec3::ZERO;
        for i in 0..steps.radial {
            // Midpoint in radius, with the disk's area element r·dr·dφ.
            let r = (i as f64 + 0.5) / steps.radial as f64;
            for j in 0..steps.angular {
                let phi = (j as f64 + 0.5) / steps.angular as f64 * std::f64::consts::TAU;
                let (a, b) = (r * phi.cos(), r * phi.sin());
                let point = centre + RADIUS * (a * u + b * w);
                let offset = point - receiver;
                let distance = offset.length();
                let direction = offset / distance;
                let cos_receiver = direction.dot(normal).max(0.0);
                let cos_emitter = (-direction).dot(toward_receiver).max(0.0);
                let area = r
                    * (1.0 / steps.radial as f64)
                    * (std::f64::consts::TAU / steps.angular as f64)
                    * RADIUS
                    * RADIUS;
                let weight = area * cos_emitter / (distance * distance) * radiance * cos_receiver;
                total += evaluate(params, normal, view, direction) * weight;
            }
        }
        total
    }

    fn native(base: [f64; 4], metallic: f64) -> BsdfParams {
        let material = Material {
            base_color: base,
            metallic,
            roughness: 0.24,
            ..Material::default()
        };
        BsdfParams::from_material(&material, DVec3::Z)
    }

    fn assert_matches_oracle(name: &str, params: &BsdfParams, mean: [f64; 3], stderr: [f64; 3]) {
        assert_matches_oracle_at(name, params, mean, stderr, COARSE);
    }

    fn assert_matches_oracle_at(
        name: &str,
        params: &BsdfParams,
        mean: [f64; 3],
        stderr: [f64; 3],
        steps: Steps,
    ) {
        let potter = probe_pixel(params, steps);
        for channel in 0..3 {
            let tolerance = 4.0 * stderr[channel];
            assert!(
                (potter[channel] - mean[channel]).abs() <= tolerance,
                "{name} channel {channel}: Potter {} vs Blender {} (tolerance {tolerance})",
                potter[channel],
                mean[channel]
            );
        }
    }

    #[test]
    fn single_bounce_metal_r024_matches_blender() {
        let params = native([0.055, 0.15, 0.24, 1.0], 1.0);
        assert_matches_oracle(
            "metal_r024",
            &params,
            [
                3.615_886_153_886_094_7,
                9.598_245_684_988_797,
                15.271_571_726_538_241,
            ],
            [
                3.895_179_372_176_688_5e-4,
                1.033_227_747_352_539_5e-3,
                1.643_885_182_480_052_4e-3,
            ],
        );
    }

    #[test]
    fn single_bounce_metal_white_matches_blender() {
        let params = native([1.0, 1.0, 1.0, 1.0], 1.0);
        assert_matches_oracle(
            "metal_r024_white",
            &params,
            [63.410_192_094_743_25; 3],
            [6.732_953_963_258_015e-3; 3],
        );
    }

    #[test]
    fn single_bounce_spec_only_matches_blender() {
        let params = native([0.0, 0.0, 0.0, 1.0], 0.0);
        assert_matches_oracle(
            "spec_only_r024",
            &params,
            [3.192_535_599_227_994_7; 3],
            [3.107_007_194_890_755e-4; 3],
        );
    }

    #[test]
    fn single_bounce_diffuse_matches_blender() {
        let mut params = native([0.8, 0.8, 0.8, 1.0], 0.0);
        params.specular_ior_level = 0.0;
        assert_matches_oracle(
            "diffuse_white",
            &params,
            [3.286_944_471_765_309_6; 3],
            [1.506_987_510_377_347_2e-4; 3],
        );
    }

    #[test]
    fn single_bounce_dielectric_white_matches_blender() {
        let params = native([0.8, 0.8, 0.8, 1.0], 0.0);
        assert_matches_oracle(
            "dielectric_white",
            &params,
            [6.313_597_380_649_298; 3],
            [9.449_430_068_234_293e-4; 3],
        );
    }

    /// Coat alone: the base diffuse is black and Specular IOR Level 0 removes the base dielectric
    /// layer, so the pixel is the coat's own single-bounce integral (`coat_r024`). Its mean equals
    /// the `spec_only_r024` oracle, as the same GGX dielectric layer with IOR 1.5 would give.
    #[test]
    fn single_bounce_coat_r024_matches_blender() {
        let mut params = native([0.0, 0.0, 0.0, 1.0], 0.0);
        params.specular_ior_level = 0.0;
        params.coat_weight = 1.0;
        params.coat_roughness = 0.24;
        params.coat_ior = 1.5;
        assert_matches_oracle_at(
            "coat_r024",
            &params,
            [3.192_535_599_227_994_7; 3],
            [3.107_007_194_890_755e-4; 3],
            FINE,
        );
    }

    /// A tinted coat over a white diffuse base: the diffuse and the default dielectric layer are
    /// attenuated by the coat's albedo and tint (`coat_tint_diffuse`).
    #[test]
    fn single_bounce_coat_tinted_over_diffuse_matches_blender() {
        let mut params = native([0.8, 0.8, 0.8, 1.0], 0.0);
        params.roughness = 0.5;
        params.coat_weight = 0.5;
        params.coat_roughness = 0.1;
        params.coat_ior = 1.5;
        params.coat_tint = [1.0, 0.5, 0.25];
        assert_matches_oracle_at(
            "coat_tint_diffuse",
            &params,
            [
                5.929_943_926_632_404,
                4.795_929_251_704_365,
                4.279_176_461_044_699,
            ],
            [
                8.170_508_334_674_762e-4,
                8.176_943_583_903_099e-4,
                8.181_972_915_078_506e-4,
            ],
            FINE,
        );
    }
}
