//! Principled and standalone BSDF kernels for the path tracer, ported from Cycles v5.2.2 (Blender 5.2.2 is
//! the oracle). Pure functions: random draws stay in `path.rs`.
//! Standalone sheen, subsurface, translucent, refraction, toon and metallic kernels use the same Cycles
//! 5.2.2 closure evaluation, sampling and continuous-density conventions.
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
//! - Standalone closures follow `bsdf_sheen.h`, `bsdf_ashikhmin_velvet.h`, `bsdf_toon.h`,
//!   `bsdf_diffuse.h`, `bsdf_microfacet.h`, `bsdf_util.h`, and `bssrdf.h`.
//!   `standalone_eval` returns the cosine-unweighted BSDF coefficient; `standalone_density` is its
//!   PDF, and `standalone_sample` distinguishes continuous, singular and subsurface draws.
//!
//! Estimator contract: path NEE evaluates each continuous BSDF times reflectance and its closure cosine: diffuse,
//! specular, sheen and coat above the surface, with glass reflection above and glass transmission below it.
//! Thin Wall's glass split uses Cycles' two-interface Fresnel weights and the transmission lobe's
//! GGX mirror model with a flipped normal; its smooth transmitted ray continues straight through.
//! `continuous_density` uses the same lobe probabilities as sampling, so `evaluate / density` is
//! unbiased for each closure. `path.rs` mixes closures by selection weight. Thick glass follows
//! Cycles' `bsdf_microfacet_eval` eval/pdf, which also covers half vectors the VNDF sampler never
//! draws, so MIS on that lobe matches Cycles' eval/pdf but is not an exact density. Singular glass
//! lobes are delta samples and cannot be reached by light sampling.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use glam::DVec3;

use super::principled_tables::{
    ggx_energy, ggx_energy_average, ggx_gen_schlick_ior_s, ggx_glass_energy,
    ggx_glass_energy_average, ggx_glass_inverse_energy, ggx_glass_inverse_energy_average,
    sheen_ltc,
};
use super::principled_thin_film::{
    THICKNESS_CUTOFF, dielectric_fresnel as thin_film_dielectric_fresnel,
    f82_fresnel as thin_film_f82_fresnel,
};
use super::principled_thin_film_table::THIN_FILM_CMF;
use super::subsurface::{channel_mask, raw_radius, subsurface_evaluated};
use crate::model::RayKind;
use crate::shader::{
    AnisotropyModel, BsdfModel, BsdfParams, GlassDistribution, MetallicFresnel, MetallicParameters,
    MicrofacetDistribution, SheenDistribution, SubsurfaceMethod, ToonComponent, ToonParameters,
    dielectric_f0, make_orthonormals,
};

/// Roughness floor shared with the sampler; Cycles treats roughness below ~1e-5 as delta.
const MIN_ROUGHNESS: f64 = 0.02;
const F64_SIGN_BIT: u64 = 1 << 63;

/// IEEE equality without a float comparison lint; NaN and signed-zero behavior match `==`.
fn exact_f64_eq(left: f64, right: f64) -> bool {
    if left.is_nan() || right.is_nan() {
        return false;
    }
    let left_bits = left.to_bits();
    let right_bits = right.to_bits();
    left_bits == right_bits || ((left_bits | right_bits) & !F64_SIGN_BIT) == 0
}

/// Selection probabilities of the sampling branches, identical to the thresholds `path.rs` uses to
/// pick a branch: subsurface, diffuse, sheen, coat, specular, transmission.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LobeProbabilities {
    pub(crate) diffuse: f64,
    pub(crate) subsurface: f64,
    pub(crate) sheen: f64,
    pub(crate) coat: f64,
    pub(crate) specular: f64,
    pub(crate) transmission: f64,
}

pub(crate) fn lobe_probabilities(params: &BsdfParams, layers: &Layers) -> LobeProbabilities {
    let metallic = params.metallic.clamp(0.0, 1.0);
    let transmission = params.transmission.clamp(0.0, 1.0);
    // The diffuse lobe keeps `base × subsurface_share` per channel (`diffuse_lobe`): a channel with no radius keeps its
    // full diffuse. The diffuse branch is chosen with the channel average of that share, so it is never starved.
    let diffuse_share = subsurface_share(params).element_sum() / 3.0;
    let diffuse_weight = ((1.0 - metallic) * (1.0 - transmission) * diffuse_share).max(0.0);
    // Thin Wall ignores the radius and splits the entire subsurface closure into reflection and transmission.
    let covered_channels = if params.thin_wall {
        1.0
    } else {
        channel_mask(raw_radius(params))
            .iter()
            .filter(|&&on| on)
            .count() as f64
    };
    let subsurface_weight = subsurface_closure_weight(
        params,
        layers.energy_roughness,
        layers.normal,
        layers.view,
        layers.lower,
        layers.caustics_reflective,
    )
    .element_sum()
        / 3.0
        * covered_channels;
    // Each layer's selection weight is its layering albedo: Cycles' sample weight of the closure.
    let sheen_weight = layers.sheen.selection_weight();
    let coat_weight = if layers.caustics_reflective {
        layers.coat.map_or(0.0, |coat| coat.albedo)
    } else {
        0.0
    };
    // A thin glass material creates two closures; their mean spectral weights determine the glass branch's selection
    // weight and the conditional reflection/transmission choice.
    let glass_weight = layers.glass.map_or(0.0, |glass| {
        (1.0 - metallic) * transmission * glass.selection_weight()
    });
    // Cycles estimates a thin-film closure's sample weight from its view-angle Fresnel instead of the roughness LUT.
    let dielectric_weight = if params.thin_film_thickness > THICKNESS_CUTOFF {
        specular_layer(params, true).map_or(0.0, |layer| {
            thin_film_dielectric_fresnel(
                layers.normal.dot(layers.view).max(0.0),
                params.thin_film_thickness,
                params.thin_film_ior.max(1.0e-5),
                layer.eta,
                layer.f0_tinted,
            )
            .0
            .element_sum()
                / 3.0
        })
    } else {
        dielectric_f0(params.ior, params.specular_ior_level)
    };
    let specular_weight = if layers.caustics_reflective {
        let weight = metallic + dielectric_weight * (1.0 - metallic) * (1.0 - transmission);
        let has_specular_eval = metallic > 0.0
            || ((1.0 - metallic) * (1.0 - transmission) > 0.0
                && specular_layer(params, true).is_some());
        // Source sampling omits a specular proposal that has no evaluation energy; legacy keeps its floor.
        if layers.source_bsdf_sampling && !has_specular_eval {
            0.0
        } else {
            weight.max(1.0e-4)
        }
    } else {
        0.0
    };
    let total = diffuse_weight
        + subsurface_weight
        + sheen_weight
        + coat_weight
        + glass_weight
        + specular_weight;
    if total <= 0.0 {
        return LobeProbabilities {
            diffuse: 0.0,
            subsurface: 0.0,
            sheen: 0.0,
            coat: 0.0,
            specular: 0.0,
            transmission: 0.0,
        };
    }
    LobeProbabilities {
        diffuse: diffuse_weight / total,
        subsurface: subsurface_weight / total,
        sheen: sheen_weight / total,
        coat: coat_weight / total,
        specular: specular_weight / total,
        transmission: glass_weight / total,
    }
}

/// One continuous lobe at a direction: its weight, the selection probability times its density there, which is
/// Cycles' `bsdf_pdf(wo) · sample_weight` of its closure, and its squared roughness, Cycles'
/// `bsdf_get_specular_roughness_squared` (zero for a singular lobe).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LobeDensity {
    pub(crate) weight: f64,
    pub(crate) roughness_squared: f64,
}

/// Roughness below which a microfacet lobe is singular (Cycles' `roughness_is_almost_specular`).
const DELTA_ROUGHNESS: f64 = 1.0e-5;

/// Cycles' `bsdf_get_specular_roughness_squared` of a microfacet lobe, `alpha_x · alpha_y` with alpha the squared
/// roughness, so roughness⁴. A singular roughness has 0.
fn microfacet_roughness_squared(roughness: f64) -> f64 {
    if roughness < DELTA_ROUGHNESS {
        0.0
    } else {
        roughness.powi(4)
    }
}

/// The continuous lobes (diffuse, thin translucent, sheen, coat, specular and glass) at `light`, in that order. Each
/// lobe's density is the one [`continuous_density`] sums: diffuse and thin translucent are cosine-distributed, sheen
/// uses LTC, coat and specular use GGX half-vector sampling, and glass is reflection above or transmission below.
pub(crate) fn continuous_lobes(
    params: &BsdfParams,
    layers: &Layers,
    light: DVec3,
) -> [LobeDensity; 6] {
    let (normal, view) = (layers.normal, layers.view);
    let probabilities = lobe_probabilities(params, layers);
    let absent = LobeDensity {
        weight: 0.0,
        roughness_squared: 0.0,
    };
    let diffuse = LobeDensity {
        weight: probabilities.diffuse * normal.dot(light).max(0.0) / PI,
        roughness_squared: 1.0,
    };
    let thin_subsurface = if params.thin_wall {
        LobeDensity {
            weight: probabilities.subsurface * (-normal.dot(light)).max(0.0) / PI,
            roughness_squared: 1.0,
        }
    } else {
        absent
    };
    let sheen = LobeDensity {
        weight: probabilities.sheen * layers.sheen.density(light),
        roughness_squared: 1.0,
    };
    let coat = layers.coat.map_or(absent, |coat| LobeDensity {
        weight: probabilities.coat
            * ndf_half_vector_pdf(
                coat.normal,
                isotropic_microfacet(coat.roughness),
                view,
                light,
            ),
        roughness_squared: microfacet_roughness_squared(coat.roughness),
    });
    let specular = LobeDensity {
        weight: probabilities.specular
            * visible_half_vector_pdf(layers.specular, layers.specular_microfacet, view, light),
        roughness_squared: layers.specular_microfacet.roughness_squared(),
    };
    let glass = layers.glass.as_ref().map_or(absent, |glass| {
        if glass.thin.is_some() {
            thin_glass_density(
                glass,
                layers.specular,
                view,
                light,
                probabilities.transmission,
            )
        } else {
            LobeDensity {
                weight: probabilities.transmission
                    * glass_eval(glass, layers.specular, view, light).1,
                // A singular glass lobe has no roughness, and Cycles' singular closure evaluates to zero.
                roughness_squared: if glass.singular {
                    0.0
                } else {
                    glass.alpha * glass.alpha
                },
            }
        }
    });
    [diffuse, thin_subsurface, sheen, coat, specular, glass]
}

/// Density of the continuous (diffuse, sheen, coat and specular) directions drawn by `path.rs`, in solid
/// angle, with the glass lobe's density as Cycles' eval and pdf define it: the sum of [`continuous_lobes`]'
/// weights.
pub(crate) fn continuous_density(params: &BsdfParams, layers: &Layers, light: DVec3) -> f64 {
    continuous_lobes(params, layers, light)
        .iter()
        .map(|lobe| lobe.weight)
        .sum()
}
/// The selected transmitted closure's `bsdf_get_specular_roughness_squared` for film transparency.
pub(crate) fn transmitted_roughness_squared(params: &BsdfParams, layers: &Layers) -> f64 {
    match params.model {
        BsdfModel::Principled => {
            let Some(glass) = layers.glass else {
                return 1.0;
            };
            if let Some(thin) = glass.thin {
                if thin.transmission_enabled {
                    thin.transmission_alpha * thin.transmission_alpha
                } else {
                    0.0
                }
            } else {
                glass.alpha * glass.alpha
            }
        }
        BsdfModel::Refraction(
            distribution @ (MicrofacetDistribution::Beckmann | MicrofacetDistribution::Ggx),
        ) => standalone_refraction_microfacet(params, layers, distribution).roughness_squared(),
        _ => 1.0,
    }
}

/// Cycles' `SD_BSDF_HAS_EVAL`: whether an active closure set has any non-singular BSDF to pair with direct-light sampling.
pub(crate) fn bsdf_has_direct_eval(params: &BsdfParams, layers: &Layers) -> bool {
    match params.model {
        BsdfModel::Holdout
        | BsdfModel::RayPortal(_)
        | BsdfModel::Refraction(MicrofacetDistribution::MultiGgx) => false,
        BsdfModel::Subsurface => {
            params.thin_wall || standalone_subsurface_weights(params).1.max_element() > 0.0
        }
        BsdfModel::Refraction(distribution) => {
            let microfacet = standalone_refraction_microfacet(params, layers, distribution);
            microfacet.alpha_x * microfacet.alpha_y > GLASS_SINGULAR_ALPHA_SQUARED
        }
        BsdfModel::Metallic(metallic) => {
            let microfacet = standalone_metallic_microfacet(params, layers, metallic.distribution);
            microfacet.alpha_x * microfacet.alpha_y > GLASS_SINGULAR_ALPHA_SQUARED
        }
        BsdfModel::Principled => {
            let normal = layers.normal;
            [normal, -normal, layers.view, -layers.view]
                .into_iter()
                .flat_map(|light| continuous_lobes(params, layers, light))
                .any(|lobe| {
                    lobe.weight > 0.0 && lobe.roughness_squared > GLASS_SINGULAR_ALPHA_SQUARED
                })
        }
        _ => true,
    }
}

/// Density of a normal drawn from the isotropic GGX NDF and reflected around the half vector.
fn ndf_half_vector_pdf(normal: DVec3, microfacet: GgxMicrofacet, view: DVec3, light: DVec3) -> f64 {
    let half = (view + light).normalize_or_zero();
    let n_dot_h = normal.dot(half).max(0.0);
    let v_dot_h = view.dot(half).max(0.0);
    if v_dot_h <= 0.0 {
        return 0.0;
    }
    microfacet.distribution(half, normal) * n_dot_h / (4.0 * v_dot_h).max(1.0e-8)
}

/// Density of a visible-normal GGX sample in solid angle, as Cycles' `bsdf_microfacet_eval` returns it.
fn visible_half_vector_pdf(
    normal: DVec3,
    microfacet: GgxMicrofacet,
    view: DVec3,
    light: DVec3,
) -> f64 {
    let half = (view + light).normalize_or_zero();
    let n_dot_v = normal.dot(view).max(0.0);
    let v_dot_h = view.dot(half);
    if n_dot_v <= 0.0 || v_dot_h <= 0.0 {
        return 0.0;
    }
    let pdf_half = microfacet.distribution(half, normal) / (1.0 + microfacet.lambda(view, normal));
    pdf_half / (4.0 * n_dot_v).max(1.0e-8)
}

/// Diffuse lobe of the Principled BSDF: Cycles' improved energy-preserving Oren-Nayar closure
/// under the dielectric-layering weight, times reflectance, less the share the subsurface closure
/// takes. The path tracer evaluates its view- and light-dependent BRDF.
pub(crate) fn diffuse_coefficient(params: &BsdfParams, layers: &Layers, light: DVec3) -> DVec3 {
    diffuse_coefficient_with_roughness(params, layers, light, params.diffuse_roughness, true)
}

fn diffuse_coefficient_with_roughness(
    params: &BsdfParams,
    layers: &Layers,
    light: DVec3,
    diffuse_roughness: f64,
    apply_thin_film: bool,
) -> DVec3 {
    diffuse_lobe(
        params,
        layers.energy_roughness,
        layers.normal,
        layers.view,
        light,
        layers.lower,
        diffuse_roughness,
        apply_thin_film,
        layers.caustics_reflective,
    )
}

/// The parameters with no subsurface closure, for the raster preview, which shades the diffuse lobe without the
/// subsurface share (the realtime gap). The path tracer does not use this.
pub(crate) fn without_subsurface(params: &BsdfParams) -> BsdfParams {
    BsdfParams {
        subsurface_weight: 0.0,
        ..*params
    }
}

/// The Lambert weight of the dielectric-layered diffuse lobe, before the 1/π and the albedo (Cycles' diffuse weight).
/// Active thin film follows Cycles' fallback to the view-angle Fresnel when the roughness LUT is unavailable.
fn diffuse_layer_weight(
    params: &BsdfParams,
    energy_roughness: f64,
    normal: DVec3,
    view: DVec3,
    apply_thin_film: bool,
    reflective_caustics: bool,
) -> f64 {
    let n_dot_v = normal.dot(view).max(0.0);
    let roughness = energy_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let metallic = params.metallic.clamp(0.0, 1.0);
    let transmission = params.transmission.clamp(0.0, 1.0);
    let dielectric_weight = ((1.0 - metallic) * (1.0 - transmission)).max(0.0);
    if !reflective_caustics {
        return dielectric_weight;
    }
    match specular_layer(params, apply_thin_film) {
        Some(layer) => {
            let albedo = if apply_thin_film && params.thin_film_thickness > THICKNESS_CUTOFF {
                thin_film_dielectric_fresnel(
                    n_dot_v,
                    params.thin_film_thickness,
                    params.thin_film_ior.max(1.0e-5),
                    layer.eta,
                    layer.f0_tinted,
                )
                .0
            } else {
                let z = ((layer.eta - 1.0) / (layer.eta + 1.0)).abs().sqrt();
                let albedo_remap = ggx_gen_schlick_ior_s(roughness, n_dot_v, z);
                layer.f0_tinted.lerp(DVec3::ONE, albedo_remap)
            };
            let energy = ggx_energy(roughness, n_dot_v).max(1.0e-6);
            let darkened = albedo * dielectric_darkening(&layer, roughness, energy);
            layered_diffuse_weight(dielectric_weight, darkened.max_element())
        }
        None => dielectric_weight,
    }
}

/// The share of the diffuse base that the subsurface closures leave to diffuse, as Cycles' `closure` setup does
/// (`svm/closure.h`). The diffuse closure keeps `1 - Subsurface Weight` for every method. Thin Wall adds its thin
/// reflection (`bsdf_thin_subsurface_setup`), a Lambert of weight `sat(0.5 (1 - anisotropy)) × subsurface weight`, to
/// the same diffuse. Otherwise `bssrdf_setup` moves a channel whose raw radius is below `BSSRDF_MIN_RADIUS` into a
/// diffuse BSDF of its closure weight, so that channel keeps its full diffuse; the other channels keep `1 - Subsurface
/// Weight`. Every method takes this share.
fn subsurface_share(params: &BsdfParams) -> DVec3 {
    let sss = params.subsurface_weight.clamp(0.0, 1.0);
    if params.thin_wall {
        return DVec3::splat(1.0 - sss + sss * thin_subsurface_reflection_share(params));
    }
    if !subsurface_evaluated(params) {
        return DVec3::ONE;
    }
    // Every method keeps its diffuse on a channel below `BSSRDF_MIN_RADIUS` (the raw radius, `bssrdf_setup`).
    let mask = channel_mask(raw_radius(params));
    DVec3::from_array(std::array::from_fn(|channel| {
        if mask[channel] { 1.0 - sss } else { 1.0 }
    }))
}

fn thin_subsurface_reflection_share(params: &BsdfParams) -> f64 {
    f64::midpoint(1.0, -params.subsurface_anisotropy).clamp(0.0, 1.0)
}

fn thin_subsurface_transmission_share(params: &BsdfParams) -> f64 {
    f64::midpoint(1.0, params.subsurface_anisotropy).clamp(0.0, 1.0)
}

/// The subsurface closure's weight per channel, `clamped base × subsurface weight × diffuse layer weight × reflectance
/// × lower`, zero on channels without a radius (Cycles' `closure_weight`, `svm/closure.h`). For Thin Wall this is the
/// translucent closure's share, which does not depend on radius.
pub(crate) fn subsurface_closure_weight(
    params: &BsdfParams,
    energy_roughness: f64,
    normal: DVec3,
    view: DVec3,
    lower: DVec3,
    reflective_caustics: bool,
) -> DVec3 {
    if !subsurface_evaluated(params) || normal.dot(view) <= 0.0 {
        return DVec3::ZERO;
    }
    let layer = diffuse_layer_weight(
        params,
        energy_roughness,
        normal,
        view,
        true,
        reflective_caustics,
    ) * params.reflectance.clamp(0.0, 1.0)
        * lower;
    if params.thin_wall {
        return base_color(params)
            * (params.subsurface_weight.clamp(0.0, 1.0)
                * thin_subsurface_transmission_share(params)
                * layer);
    }
    let mask = channel_mask(raw_radius(params));
    let covered = DVec3::from_array(mask.map(|on| if on { 1.0 } else { 0.0 }));
    base_color(params) * covered * (params.subsurface_weight.clamp(0.0, 1.0) * layer)
}

/// The diffuse lobe, given the coat's lower-layer factor (set up once by the caller).
fn diffuse_lobe(
    params: &BsdfParams,
    energy_roughness: f64,
    normal: DVec3,
    view: DVec3,
    light: DVec3,
    lower: DVec3,
    diffuse_roughness: f64,
    apply_thin_film: bool,
    reflective_caustics: bool,
) -> DVec3 {
    let n_dot_v = normal.dot(view).max(0.0);
    if n_dot_v <= 0.0 {
        return DVec3::ZERO;
    }
    let diffuse_color = if params.thin_wall {
        let sss = params.subsurface_weight.clamp(0.0, 1.0);
        diffuse_albedo(params) * (1.0 - sss)
            + base_color(params) * (sss * thin_subsurface_reflection_share(params))
    } else {
        diffuse_albedo(params) * subsurface_share(params)
    };
    diffuse_color
        * oren_nayar_brdf(base_color(params), diffuse_roughness, normal, view, light)
        * diffuse_layer_weight(
            params,
            energy_roughness,
            normal,
            view,
            apply_thin_film,
            reflective_caustics,
        )
        * params.reflectance.clamp(0.0, 1.0)
        * lower
}
fn oren_nayar_brdf(
    color: DVec3,
    roughness: f64,
    normal: DVec3,
    view: DVec3,
    light: DVec3,
) -> DVec3 {
    let sigma = roughness.clamp(0.0, 1.0);
    if sigma < 1.0e-5 {
        return DVec3::splat(1.0 / PI);
    }
    let a = 1.0 / (PI + sigma * (FRAC_PI_2 - 2.0 / 3.0));
    let b = sigma * a;
    let nl = normal.dot(light).max(0.0);
    let nv = normal.dot(view).max(0.0);
    let mut t = light.dot(view) - nl * nv;
    if t > 0.0 {
        t /= nl.max(nv) + f64::MIN_POSITIVE;
    }
    let single_scatter = a + b * t;
    let e_avg = a * PI + ((2.0 * PI - 5.6) / 3.0) * b;
    let albedo = color.clamp(DVec3::ZERO, DVec3::ONE).to_array();
    let multi_scatter = DVec3::from_array(albedo.map(|channel| {
        channel * channel * (e_avg / (1.0 - e_avg)) / PI / (1.0 - channel * (1.0 - e_avg))
    }));
    let e_v = a * PI + b * oren_nayar_g(nv);
    let e_l = a * PI + b * oren_nayar_g(nl);
    DVec3::splat(single_scatter) + multi_scatter * ((1.0 - e_v) * (1.0 - e_l))
}

fn oren_nayar_g(cos_theta: f64) -> f64 {
    let cos_theta = cos_theta.clamp(0.0, 1.0);
    if cos_theta < 1.0e-6 {
        return FRAC_PI_2 - 2.0 / 3.0 - cos_theta;
    }
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let theta = cos_theta.acos();
    sin_theta * (theta - 2.0 / 3.0 - sin_theta * cos_theta)
        + 2.0 / 3.0 * (sin_theta / cos_theta) * (1.0 - sin_theta * sin_theta * sin_theta)
}

/// Specular (dielectric and metallic) lobes along `light`, times reflectance. Zero when the
/// light or view is below the surface.
pub(crate) fn specular_coefficient(params: &BsdfParams, layers: &Layers, light: DVec3) -> DVec3 {
    specular_lobe(
        params,
        layers.energy_roughness,
        layers.specular_microfacet,
        layers.specular,
        layers.view,
        light,
        layers.lower,
        true,
    )
}

/// The specular lobe, given the coat's lower-layer factor (set up once by the caller).
/// `apply_thin_film` is false for EEVEE's realtime lobe path, which ignores these sockets.
fn specular_lobe(
    params: &BsdfParams,
    energy_roughness: f64,
    microfacet: GgxMicrofacet,
    normal: DVec3,
    view: DVec3,
    light: DVec3,
    lower: DVec3,
    apply_thin_film: bool,
) -> DVec3 {
    let n_dot_v = normal.dot(view).max(0.0);
    let n_dot_l = normal.dot(light).max(0.0);
    if n_dot_v <= 0.0 || n_dot_l <= 0.0 {
        return DVec3::ZERO;
    }
    let half = (view + light).normalize_or_zero();
    let v_dot_h = view.dot(half).max(0.0);
    let metallic = params.metallic.clamp(0.0, 1.0);
    let transmission = params.transmission.clamp(0.0, 1.0);
    let base = base_color(params);

    // Energy preservation: E is looked up at the view direction, as Cycles does at setup
    // (`microfacet_ggx_preserve_energy(kg, bsdf, sd->wi, ...)`), so it is constant per shading point.
    // The compensation is Cycles' setup value (`energy_scale`), which the glossy blur leaves at the original roughness.
    let energy_roughness = energy_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let energy = ggx_energy(energy_roughness, n_dot_v).max(1.0e-6);
    let energy_average = ggx_energy_average(energy_roughness);
    let missing = (1.0 - energy) / energy;

    let common = microfacet.distribution(half, normal)
        / (1.0 + microfacet.lambda(view, normal) + microfacet.lambda(light, normal))
        / (4.0 * n_dot_v * n_dot_l).max(1.0e-8);

    let dielectric_weight = ((1.0 - metallic) * (1.0 - transmission)).max(0.0);
    let mut reflected = DVec3::ZERO;

    if let Some(layer) = specular_layer(params, apply_thin_film) {
        let f0 = layer.f0_tinted;
        let fresnel = if apply_thin_film && params.thin_film_thickness > THICKNESS_CUTOFF {
            thin_film_dielectric_fresnel(
                v_dot_h,
                params.thin_film_thickness,
                params.thin_film_ior.max(1.0e-5),
                layer.eta,
                f0,
            )
            .0
        } else {
            let real_f = fresnel_dielectric(v_dot_h, layer.eta);
            remapped_fresnel(f0, layer.eta, real_f)
        };
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
        let fresnel = if apply_thin_film && params.thin_film_thickness > THICKNESS_CUTOFF {
            thin_film_f82_fresnel(
                v_dot_h,
                params.thin_film_thickness,
                params.thin_film_ior.max(1.0e-5),
                base,
                b,
            )
        } else {
            f82_fresnel(v_dot_h, base, b)
        };
        let fss = f82_fss(base, b);
        let gain = multiscatter_gain(fss, energy_average, missing);
        reflected += gain * fresnel * metallic;
    }

    (reflected * common) * (params.reflectance.clamp(0.0, 1.0) * lower)
}

/// Cycles' `CLOSURE_WEIGHT_CUTOFF`: a coat weight at or below it is not set up.
const COAT_WEIGHT_CUTOFF: f64 = 1.0e-5;

/// Cycles' Thin Wall translucent closure's lobe weight before its Oren-Nayar evaluation.
pub(crate) fn thin_subsurface_transmission_weight(
    params: &BsdfParams,
    layers: &Layers,
    apply_thin_film: bool,
) -> DVec3 {
    let (normal, view) = (layers.normal, layers.view);
    if !params.thin_wall || !subsurface_evaluated(params) || normal.dot(view) <= 0.0 {
        return DVec3::ZERO;
    }
    DVec3::splat(
        params.subsurface_weight.clamp(0.0, 1.0)
            * thin_subsurface_transmission_share(params)
            * diffuse_layer_weight(
                params,
                layers.energy_roughness,
                normal,
                view,
                apply_thin_film,
                layers.caustics_reflective,
            )
            * params.reflectance.clamp(0.0, 1.0),
    ) * layers.lower
}
/// Cycles' thin-subsurface translucent closure on the transmitted hemisphere.
pub(crate) fn thin_subsurface_transmission_coefficient(
    params: &BsdfParams,
    layers: &Layers,
    light: DVec3,
    diffuse_roughness: f64,
    apply_thin_film: bool,
) -> DVec3 {
    let (normal, view) = (layers.normal, layers.view);
    if normal.dot(light) >= 0.0 {
        return DVec3::ZERO;
    }
    let virtual_view = view - normal * (2.0 * normal.dot(view));
    base_color(params)
        * thin_subsurface_transmission_weight(params, layers, apply_thin_film)
        * oren_nayar_brdf(
            base_color(params),
            diffuse_roughness,
            -normal,
            virtual_view,
            light,
        )
}

/// EEVEE's `ClosureTranslucent` uses the unclamped color and ignores Diffuse Roughness.
fn eevee_thin_subsurface_transmission_coefficient(
    params: &BsdfParams,
    layers: &Layers,
    light: DVec3,
) -> DVec3 {
    let (normal, view) = (layers.normal, layers.view);
    if normal.dot(light) >= 0.0 {
        return DVec3::ZERO;
    }
    let virtual_view = view - normal * (2.0 * normal.dot(view));
    diffuse_albedo(params)
        * thin_subsurface_transmission_weight(params, layers, false)
        * oren_nayar_brdf(base_color(params), 0.0, -normal, virtual_view, light)
}

/// The sheen lobe along `light`, times reflectance. It depends on the light, so an area light is
/// integrated numerically or at its centre (`scene.rs`), like the specular lobe.
pub(crate) fn sheen_coefficient(params: &BsdfParams, layers: &Layers, light: DVec3) -> DVec3 {
    layers.sheen.value(light) * params.reflectance.clamp(0.0, 1.0)
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

/// The coat layer for `view`, or `None` when the coat is off or the view is below its normal. `normal` is the coat's
/// normal: the linked Coat Normal, or the final shading normal when unlinked, bent as Cycles' coat is (`layers_at`).
fn coat_layer(params: &BsdfParams, normal: DVec3, view: DVec3) -> Option<CoatLayer> {
    let weight = params.coat_weight.max(0.0);
    if weight <= COAT_WEIGHT_CUTOFF {
        return None;
    }
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

/// The anisotropic GGX axes and local frame shared by a closure's evaluation, pdf and sampler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GgxMicrofacet {
    alpha_x: f64,
    alpha_y: f64,
    tangent: DVec3,
    bitangent: DVec3,
    anisotropic: bool,
}

impl GgxMicrofacet {
    fn local(self, vector: DVec3, normal: DVec3) -> DVec3 {
        DVec3::new(
            self.tangent.dot(vector),
            self.bitangent.dot(vector),
            normal.dot(vector),
        )
    }

    fn distribution(self, half: DVec3, normal: DVec3) -> f64 {
        if self.anisotropic {
            ggx_anisotropic_distribution(self.alpha_x, self.alpha_y, self.local(half, normal))
        } else {
            ggx_distribution(self.alpha_x * self.alpha_x, normal.dot(half))
        }
    }

    fn lambda(self, direction: DVec3, normal: DVec3) -> f64 {
        if self.anisotropic {
            ggx_anisotropic_lambda(self.alpha_x, self.alpha_y, self.local(direction, normal))
        } else {
            ggx_lambda(self.alpha_x * self.alpha_x, normal.dot(direction))
        }
    }

    fn energy_roughness(self) -> f64 {
        (self.alpha_x * self.alpha_y).sqrt().sqrt()
    }

    fn blurred(mut self, blur_alpha: f64) -> Self {
        self.alpha_x = self.alpha_x.max(blur_alpha);
        self.alpha_y = self.alpha_y.max(blur_alpha);
        self
    }

    fn roughness_squared(self) -> f64 {
        self.alpha_x * self.alpha_y
    }
}

/// The layers above the base lobes at one shading point, set up once and shared by every lobe they
/// attenuate. `lower` is the factor they leave to the diffuse, specular and metallic lobes, the
/// transmission delta and the emission: the coat's layering and tint, times the sheen's layering,
/// as Cycles layers them (`principled_bsdf_emission`, `kernel/svm/closure.h`).
#[derive(Clone, Copy)]
pub(crate) struct Layers {
    /// The roughness Cycles' setup computed the multiscatter energy from (`microfacet_ggx_preserve_energy`, once per
    /// closure). It is kept through the glossy blur, which raises only the sampling alpha (`bsdf_microfacet_blur`),
    /// so a blurred closure keeps the compensation of its original roughness, as Cycles' `energy_scale` does.
    pub(crate) energy_roughness: f64,
    /// The shading normal, which the diffuse and sheen lobes use.
    pub(crate) normal: DVec3,
    /// The sheen's normal: the shading normal mixed towards the coat normal by the coat weight (`principled_bsdf_emission`).
    pub(crate) sheen_normal: DVec3,
    /// The normal of the specular and glass microfacets: the shading normal bent by [`valid_specular_normal`].
    pub(crate) specular: DVec3,
    /// Principled reflection/metal axes. Cycles 5.2.2 keeps transmission isotropic.
    pub(crate) specular_microfacet: GgxMicrofacet,
    /// The coat's normal, bent the same way as Cycles' `valid_coat_normal`.
    pub(crate) coat_normal: DVec3,
    pub(crate) view: DVec3,
    coat: Option<CoatLayer>,
    sheen: SheenLobe,
    pub(crate) lower: DVec3,
    glass: Option<GlassLobe>,
    /// Cycles' glossy blur alpha (`bsdf_blur`) for standalone microfacets and principled hair; zero when unblurred.
    pub(crate) glossy_blur_alpha: f64,
    pub(crate) caustics_reflective: bool,
    /// Source-volume BSDF sampling omits dielectric proposals that have no evaluation energy.
    source_bsdf_sampling: bool,
    /// Cycles' converted `shadow_terminator_shading_offset` frequency multiplier.
    pub(crate) shadow_terminator_frequency_multiplier: f64,
}

/// Sets up the sheen and coat layers of a shading point with normal `normal`, seen along `view`. The geometric normal
/// is taken as `normal` itself, so nothing is bent. Use [`layers_at`] where the geometric normal is known.
pub(crate) fn layers(params: &BsdfParams, normal: DVec3, view: DVec3) -> Layers {
    // The normals coincide, so `use_bump_map_correction` has nothing to bend and takes its default.
    layers_at(params, normal, view, normal, true)
}

/// `layers` at a point whose geometric normal is `geometric` (Cycles' `sd->Ng`). The specular, coat and glass
/// microfacets take their normals bent by [`valid_specular_normal`]; diffuse and sheen keep `normal`.
pub(crate) fn layers_at(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    geometric: DVec3,
    bump_correction: bool,
) -> Layers {
    let specular = valid_specular_normal(geometric, normal, view, bump_correction);
    let rotation_normal = match params.anisotropy_model {
        AnisotropyModel::Principled => normal,
        AnisotropyModel::Glossy => specular,
    };
    let specular_microfacet = ggx_microfacet(params, params.roughness, rotation_normal, specular);
    let energy_roughness = specular_microfacet.energy_roughness();
    let coat_normal = valid_specular_normal(
        geometric,
        params.coat_normal_or_shading(),
        view,
        bump_correction,
    );
    let coat = coat_layer(params, coat_normal, view);
    // Cycles sets the sheen's normal to `safe_normalize(mix(N, coat normal, saturate(coat weight)))`
    // (`kernel/svm/closure.h`, `principled_bsdf_emission`), so the sheen frame and LTC follow the coat.
    let coat_weight = params.coat_weight.clamp(0.0, 1.0);
    let mixed = normal.lerp(params.coat_normal_or_shading(), coat_weight);
    let sheen_normal = if mixed.length_squared() > f64::EPSILON {
        mixed.normalize()
    } else {
        normal
    };
    let sheen = sheen_lobe(params, sheen_normal, view);
    let lower = coat_lower_factor_of(params, coat.as_ref()) * sheen.layer_weight();
    let glass = glass_lobe(params, specular, view, true, true);
    Layers {
        energy_roughness,
        normal,
        sheen_normal,
        specular,
        specular_microfacet,
        coat_normal,
        view,
        coat,
        sheen,
        lower,
        glass,
        glossy_blur_alpha: 0.0,
        caustics_reflective: true,
        source_bsdf_sampling: false,
        shadow_terminator_frequency_multiplier: 1.0,
    }
}

/// Cycles' `maybe_ensure_valid_specular_reflection` (`kernel/closure/bsdf_util.h:399-409`): `normal`, unless the material's
/// `use_bump_map_correction` is off or `normal` is the geometric normal. Cycles applies it to every closure that
/// reflects specularly (the Principled specular, coat and glass, and the Glossy and Glass nodes).
pub(crate) fn valid_specular_normal(
    geometric: DVec3,
    normal: DVec3,
    view: DVec3,
    bump_correction: bool,
) -> DVec3 {
    if !bump_correction || normal == geometric {
        return normal;
    }
    ensure_valid_specular_reflection(geometric, view, normal)
}

/// Cycles' `ensure_valid_specular_reflection` (`bsdf_util.h:322-395`): raises `normal` towards `geometric` until the
/// reflection of `incoming` stays above the threshold `min(0.9 cos_i, 0.01)`. Cycles solves the reflection equation in
/// the plane through `geometric` and `normal`, where `x` is the normal's tangent and `nz` its height above `geometric`.
fn ensure_valid_specular_reflection(geometric: DVec3, incoming: DVec3, normal: DVec3) -> DVec3 {
    let reflected = 2.0 * normal.dot(incoming) * normal - incoming;
    let iz = incoming.dot(geometric);
    let threshold = (0.9 * iz).min(0.01);
    if geometric.dot(reflected) >= threshold {
        return normal;
    }
    let tangent = normal - normal.dot(geometric) * geometric;
    let x = if tangent.length_squared() > 0.0 {
        tangent.normalize()
    } else {
        normal
    };
    let ix = incoming.dot(x);
    let a = ix * ix + iz * iz;
    if a <= 0.0 {
        return normal;
    }
    let b = 2.0 * (a + iz * threshold);
    let c = (threshold + iz) * (threshold + iz);
    let root = (b * b - 4.0 * a * c).max(0.0).sqrt();
    let nz2 = if ix < 0.0 {
        0.25 * (b + root) / a
    } else {
        0.25 * (b - root) / a
    };
    let nx = (1.0 - nz2).max(0.0).sqrt();
    let nz = nz2.max(0.0).sqrt();
    nx * x + nz * geometric
}

impl Layers {
    /// Applies Cycles' incoming-diffuse masks, retaining coat attenuation and any prior Filter Glossy alpha.
    pub(crate) fn with_caustics(
        &self,
        params: &BsdfParams,
        reflective: bool,
        refractive: bool,
    ) -> Self {
        let mut layers = *self;
        layers.caustics_reflective = reflective;
        if !reflective && let Some(coat) = &mut layers.coat {
            coat.albedo = 0.0;
        }
        let mut filtered = *params;
        if !reflective {
            filtered.glass_reflection_tint = [0.0; 3];
        }
        if !refractive {
            filtered.glass_transmission_tint = [0.0; 3];
        }
        layers.glass = glass_lobe(
            &filtered,
            layers.specular,
            layers.view,
            reflective,
            refractive,
        );
        if layers.glossy_blur_alpha > 0.0
            && let Some(glass) = &mut layers.glass
        {
            glass.alpha = glass.alpha.max(layers.glossy_blur_alpha);
            glass.singular = glass.alpha * glass.alpha <= GLASS_SINGULAR_ALPHA_SQUARED;
            if let Some(thin) = &mut glass.thin {
                thin.transmission_alpha = thin.transmission_alpha.max(layers.glossy_blur_alpha);
            }
        }
        layers.lower =
            coat_lower_factor_of(params, layers.coat.as_ref()) * layers.sheen.layer_weight();
        layers
    }
    pub(crate) fn with_source_bsdf_sampling(&self) -> Self {
        let mut layers = *self;
        layers.source_bsdf_sampling = true;
        layers
    }
    /// Applies Cycles' object shading-terminator frequency to this closure.
    pub(crate) fn with_shadow_terminator_shading_offset(&self, offset: f64) -> Self {
        let mut layers = *self;
        let offset = offset as f32;
        layers.shadow_terminator_frequency_multiplier =
            f64::from(1.0_f32 / (1.0_f32 - 0.5_f32 * offset));
        layers
    }

    pub(crate) fn shadow_terminator_multiplier(&self, cos_in: f64) -> f64 {
        let frequency_multiplier = self.shadow_terminator_frequency_multiplier;
        if frequency_multiplier <= 1.0 || cos_in < 0.0 {
            return 1.0;
        }
        let cos_in = cos_in.min(1.0);
        if cos_in <= f64::EPSILON {
            return 0.0;
        }
        (cos_in.acos() * frequency_multiplier).cos().max(0.0) / cos_in
    }
    /// The closure after Cycles' glossy blur: widen cached microfacets and refresh glass singularity, retaining setup energy.
    pub(crate) fn blurred(&self, blur_alpha: f64) -> Self {
        let mut layers = *self;
        layers.specular_microfacet = layers.specular_microfacet.blurred(blur_alpha);
        layers.glossy_blur_alpha = layers.glossy_blur_alpha.max(blur_alpha);
        if let Some(coat) = &mut layers.coat {
            coat.roughness = coat.roughness.powi(2).max(blur_alpha).sqrt();
        }
        if let Some(glass) = &mut layers.glass {
            glass.alpha = glass.alpha.max(blur_alpha);
            glass.singular = glass.alpha * glass.alpha <= GLASS_SINGULAR_ALPHA_SQUARED;
            if let Some(thin) = &mut glass.thin {
                thin.transmission_alpha = thin.transmission_alpha.max(blur_alpha);
            }
        }
        layers
    }

    /// The factor the layers leave to the lobes below them and to the emission.
    pub(crate) fn lower(&self) -> DVec3 {
        self.lower
    }

    /// Sheen direction for the square sample `(u, v)` in `[0, 1)²`, drawn from the LTC distribution.
    pub(crate) fn sample_sheen(&self, u: f64, v: f64) -> DVec3 {
        self.sheen.direction(u, v)
    }
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

/// The coat lobe along `light`, times reflectance, attenuated by the sheen layer above it (Cycles
/// sets the sheen up first, so the coat's weight is the sheen-layered one). Zero when the coat is off
/// or the light is below the coat's normal.
pub(crate) fn coat_coefficient(params: &BsdfParams, layers: &Layers, light: DVec3) -> DVec3 {
    if !layers.caustics_reflective {
        return DVec3::ZERO;
    }
    coat_term(params, layers.coat.as_ref(), layers.view, light) * layers.sheen.layer_weight()
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

/// Cycles' `CLOSURE_WEIGHT_CUTOFF`: a sheen weight at or below it is not set up at all.
const CLOSURE_WEIGHT_CUTOFF: f64 = 1.0e-5;

/// The sheen (microfiber) lobe, set up as Cycles' `bsdf_sheen_setup`. The distribution is the LTC
/// transform of a cosine lobe in the frame `(tangent, bitangent, normal)` built from the view.
/// `weight` is the lobe's reflectance per channel, `tint · sheen weight · LTC albedo`. It is zero
/// when the lobe is off: sheen weight at most the closure cutoff, view below the surface, or an
/// invalid LTC, which Cycles drops.
#[derive(Clone, Copy, Debug)]
struct SheenLobe {
    normal: DVec3,
    tangent: DVec3,
    bitangent: DVec3,
    transform_a: f64,
    transform_b: f64,
    weight: DVec3,
}

fn sheen_lobe(params: &BsdfParams, normal: DVec3, view: DVec3) -> SheenLobe {
    let off = SheenLobe {
        normal,
        tangent: DVec3::ZERO,
        bitangent: DVec3::ZERO,
        transform_a: 0.0,
        transform_b: 0.0,
        weight: DVec3::ZERO,
    };
    let sheen_weight = params.sheen_weight.max(0.0);
    let cos_ni = normal.dot(view);
    if sheen_weight <= CLOSURE_WEIGHT_CUTOFF || cos_ni <= 0.0 {
        return off;
    }
    let roughness = saturate(params.sheen_roughness).clamp(1.0e-3, 1.0);
    let ltc = sheen_ltc(roughness, cos_ni);
    if ltc.transform_a.abs() < 1.0e-5 || ltc.albedo < 1.0e-5 {
        return off;
    }
    let (tangent, bitangent) = sheen_frame(normal, view);
    let tint = DVec3::from_array(params.sheen_tint).max(DVec3::ZERO);
    SheenLobe {
        normal,
        tangent,
        bitangent,
        transform_a: ltc.transform_a,
        transform_b: ltc.transform_b,
        weight: tint * (sheen_weight * ltc.albedo),
    }
}

impl SheenLobe {
    /// The lobe's share of the sampler's branch choice: its mean reflectance over the channels.
    fn selection_weight(&self) -> f64 {
        self.weight.element_sum() / 3.0
    }

    /// The weight the layers below keep, `closure_layering_weight` of the sheen albedo:
    /// `1 - max_c(sheen weight · tint_c · albedo)`.
    fn layer_weight(&self) -> f64 {
        saturate(1.0 - self.weight.max_element())
    }

    /// The LTC cosine `z` and distribution shape `(a / |L|²)²` of `light` in the lobe frame.
    fn shape(&self, light: DVec3) -> (f64, f64) {
        let local = DVec3::new(
            light.dot(self.tangent),
            light.dot(self.bitangent),
            light.dot(self.normal),
        );
        let (a, b) = (self.transform_a, self.transform_b);
        let length_squared =
            (a * local.x + b * local.z).powi(2) + (a * local.y).powi(2) + local.z.powi(2);
        (local.z, (a / length_squared).powi(2))
    }

    /// The BSDF value `weight · (a / |L|²)² / π`, Cycles' `bsdf_sheen_eval` without its cosine.
    fn value(&self, light: DVec3) -> DVec3 {
        if self.weight.max_element() <= 0.0 {
            return DVec3::ZERO;
        }
        let (cosine, shape) = self.shape(light);
        if cosine <= 0.0 {
            return DVec3::ZERO;
        }
        self.weight * (shape / PI)
    }

    /// Solid-angle density `z · (a / |L|²)² / π` of `light`, the pdf Cycles reports for it.
    fn density(&self, light: DVec3) -> f64 {
        if self.weight.max_element() <= 0.0 {
            return 0.0;
        }
        let (cosine, shape) = self.shape(light);
        if cosine <= 0.0 {
            return 0.0;
        }
        cosine * shape / PI
    }

    /// Cycles' `bsdf_sheen_sample`: the LTC-warped disk sample, in the lobe frame.
    fn direction(&self, u: f64, v: f64) -> DVec3 {
        let (disk_x, disk_y) = concentric_disk(u, v);
        let disk_z = (1.0 - disk_x * disk_x - disk_y * disk_y).max(0.0).sqrt();
        let local = DVec3::new(
            disk_x - disk_z * self.transform_b,
            disk_y,
            disk_z * self.transform_a,
        )
        .normalize_or_zero();
        self.tangent * local.x + self.bitangent * local.y + self.normal * local.z
    }
}

/// Cycles' `sample_uniform_disk`: the concentric mapping of `[0, 1)²` onto the unit disk.
pub(super) fn concentric_disk(u: f64, v: f64) -> (f64, f64) {
    let a = 2.0 * u - 1.0;
    let b = 2.0 * v - 1.0;
    if a * a + b * b <= 0.0 {
        return (0.0, 0.0);
    }
    let (radius, phi) = if a * a > b * b {
        (a, FRAC_PI_4 * (b / a))
    } else {
        (b, FRAC_PI_2 - FRAC_PI_4 * (a / b))
    };
    (radius * phi.cos(), radius * phi.sin())
}

/// Cycles' `make_orthonormals_safe_tangent(N, wi)`: the bitangent is `normalize(N × wi)` and the
/// tangent `B × N`. Only a view along the normal fails the normalisation; the basis then comes from
/// `make_orthonormals`.
fn sheen_frame(normal: DVec3, view: DVec3) -> (DVec3, DVec3) {
    let bitangent = normal.cross(view).normalize_or_zero();
    if bitangent.length_squared() < 0.99 {
        return make_orthonormals(normal);
    }
    (bitangent.cross(normal), bitangent)
}

/// The reflected lobes that EEVEE's realtime raster admits, without the glass lobe or shading-side
/// gate. EEVEE ignores Principled Diffuse Roughness, so its diffuse lobe stays Lambertian.
pub(crate) fn reflected_lobes_coefficient(
    params: &BsdfParams,
    layers: &Layers,
    light: DVec3,
    ray_mask: u8,
) -> DVec3 {
    let layers = Layers {
        energy_roughness: params.roughness,
        ..*layers
    };
    let diffuse = if ray_mask & RayKind::Diffuse.bit() != 0 {
        diffuse_coefficient_with_roughness(params, &layers, light, 0.0, false)
            + sheen_coefficient(params, &layers, light)
    } else {
        DVec3::ZERO
    };
    let coat = if ray_mask & RayKind::Glossy.bit() != 0 {
        coat_coefficient(params, &layers, light)
    } else {
        DVec3::ZERO
    };
    let specular = if ray_mask & RayKind::Glossy.bit() != 0 {
        specular_lobe(
            params,
            params.roughness,
            isotropic_microfacet(params.roughness),
            layers.specular,
            layers.view,
            light,
            layers.lower,
            false,
        )
    } else {
        DVec3::ZERO
    };
    let translucent = if ray_mask & RayKind::Diffuse.bit() != 0 {
        eevee_thin_subsurface_transmission_coefficient(params, &layers, light)
    } else {
        DVec3::ZERO
    };
    diffuse + coat + specular + translucent
}

/// Glass closure ray masks: thick glass is both glossy and transmissive, while Thin Wall's two closures have separate
/// masks (reflection above, transmission below).
pub(crate) fn glass_lobes_coefficient(
    params: &BsdfParams,
    layers: &Layers,
    light: DVec3,
    ray_mask: u8,
) -> DVec3 {
    let mask = if layers.glass.is_some_and(|glass| glass.thin.is_some()) {
        if layers.specular.dot(light) < 0.0 {
            RayKind::Transmission.bit()
        } else {
            RayKind::Glossy.bit()
        }
    } else {
        RayKind::Glossy.bit() | RayKind::Transmission.bit()
    };
    if ray_mask & mask != 0 {
        glass_coefficient(params, layers, light)
    } else {
        DVec3::ZERO
    }
}

/// The base colour clamped to 1 (Cycles' `clamped_base_color`, used by the F82 metal lobe).
pub(super) fn base_color(params: &BsdfParams) -> DVec3 {
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

/// The IOR that Cycles refracts at, after the Specular IOR Level (`svm/closure.h`, `eta`): the Principled IOR at level
/// one half, and otherwise the IOR whose F0 is the level's F0, inverted for an IOR below one. The random walk entry uses
/// it as `bssrdf->ior`.
#[expect(
    clippy::float_cmp,
    reason = "Cycles compares the specular IOR level with exactly 0.5"
)]
pub(super) fn specular_eta(params: &BsdfParams) -> f64 {
    let level = params.specular_ior_level.max(0.0);
    if level == 0.5 {
        return params.ior;
    }
    let eta = ior_from_f0(dielectric_f0(params.ior, level));
    if params.ior < 1.0 { 1.0 / eta } else { eta }
}

/// Cycles' specular setup: `f0 = F0(ior) * 2 * level`, `eta` from [`specular_eta`]. The layer exists when `eta != 1`, or when film is active.
#[expect(
    clippy::float_cmp,
    reason = "Cycles compares eta with exactly 1 when no active film requires the layer"
)]
fn specular_layer(params: &BsdfParams, apply_thin_film: bool) -> Option<SpecularLayer> {
    let level = params.specular_ior_level.max(0.0);
    let f0_scalar = dielectric_f0(params.ior, level);
    let eta = specular_eta(params);
    if eta == 1.0 && !(apply_thin_film && params.thin_film_thickness > THICKNESS_CUTOFF) {
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

fn isotropic_microfacet(roughness: f64) -> GgxMicrofacet {
    let alpha = roughness.clamp(MIN_ROUGHNESS, 1.0).powi(2);
    GgxMicrofacet {
        alpha_x: alpha,
        alpha_y: alpha,
        tangent: DVec3::ZERO,
        bitangent: DVec3::ZERO,
        anisotropic: false,
    }
}

fn ggx_microfacet(
    params: &BsdfParams,
    roughness: f64,
    rotation_normal: DVec3,
    normal: DVec3,
) -> GgxMicrofacet {
    let Some(tangent) = params.anisotropic_tangent else {
        return isotropic_microfacet(roughness);
    };
    let alpha = roughness.clamp(MIN_ROUGHNESS, 1.0).powi(2);
    let (alpha_x, alpha_y) = match params.anisotropy_model {
        AnisotropyModel::Principled => {
            let anisotropy = params.anisotropy.clamp(0.0, 1.0);
            let aspect = (1.0 - 0.9 * anisotropy).sqrt();
            (alpha / aspect, alpha * aspect)
        }
        AnisotropyModel::Glossy => {
            let anisotropy = params.anisotropy.clamp(-0.99, 0.99);
            if anisotropy < 0.0 {
                (alpha / (1.0 + anisotropy), alpha * (1.0 + anisotropy))
            } else {
                (alpha * (1.0 - anisotropy), alpha / (1.0 - anisotropy))
            }
        }
    };
    let alpha_x = alpha_x.clamp(0.0, 1.0);
    let alpha_y = alpha_y.clamp(0.0, 1.0);
    let tangent = rotate_around_axis(
        tangent,
        rotation_normal,
        params.anisotropic_rotation * 2.0 * PI,
    );
    let bitangent = normal.cross(tangent).normalize_or_zero();
    if bitangent.length_squared() <= f64::EPSILON {
        return isotropic_microfacet(roughness);
    }
    GgxMicrofacet {
        alpha_x,
        alpha_y,
        tangent: bitangent.cross(normal),
        bitangent,
        anisotropic: true,
    }
}

fn rotate_around_axis(vector: DVec3, axis: DVec3, angle: f64) -> DVec3 {
    let (sin, cos) = angle.sin_cos();
    vector * cos + axis.cross(vector) * sin + axis * (axis.dot(vector) * (1.0 - cos))
}

fn ggx_anisotropic_distribution(alpha_x: f64, alpha_y: f64, half: DVec3) -> f64 {
    let scaled = DVec3::new(half.x / alpha_x, half.y / alpha_y, half.z);
    1.0 / (PI * alpha_x * alpha_y * scaled.length_squared().powi(2)).max(1.0e-12)
}

fn ggx_anisotropic_lambda(alpha_x: f64, alpha_y: f64, direction: DVec3) -> f64 {
    let z_squared = direction.z * direction.z;
    if z_squared <= 1.0e-16 {
        return f64::INFINITY;
    }
    let alpha_squared_tangent_squared =
        ((alpha_x * direction.x).powi(2) + (alpha_y * direction.y).powi(2)) / z_squared;
    0.5 * ((1.0 + alpha_squared_tangent_squared).sqrt() - 1.0)
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

/// Cycles' `generalized_schlick_fresnel` remap (`exponent < 0`): the real dielectric Fresnel `real` at the IOR
/// `eta` is rescaled from `F0(eta)..1` onto `f0..1`.
fn remapped_fresnel(f0: DVec3, eta: f64, real: f64) -> DVec3 {
    let remap = saturate(inverse_lerp(dielectric_f0(eta, 0.5), 1.0, real));
    f0.lerp(DVec3::ONE, remap)
}

fn ior_from_f0(f0: f64) -> f64 {
    let root = f0.clamp(0.0, 0.99).sqrt();
    (1.0 + root) / (1.0 - root)
}

/// Cycles' `fresnel_dielectric_polarized` averaged over the two polarisations: the reflectance, and the
/// cosine of the refracted angle (negative, as Cycles gives it; zero under total internal reflection).
#[expect(
    clippy::manual_midpoint,
    reason = "Cycles averages the s and p reflectances as 0.5 * (s + p), so the same expression is kept"
)]
fn dielectric_fresnel(cosine_incident: f64, eta: f64) -> (f64, f64) {
    let cos_i = cosine_incident.abs();
    let eta_cos_t_squared = eta * eta - (1.0 - cos_i * cos_i);
    if eta_cos_t_squared <= 0.0 {
        return (1.0, 0.0);
    }
    let cos_t = -eta_cos_t_squared.sqrt() / eta;
    let r_s = (cos_i + eta * cos_t) / (cos_i - eta * cos_t);
    let r_p = (cos_t + eta * cos_i) / (eta * cos_i - cos_t);
    (0.5 * (r_s * r_s + r_p * r_p), cos_t)
}

/// Unpolarised dielectric Fresnel reflectance (`fresnel_dielectric`, average of s and p).
fn fresnel_dielectric(cosine_incident: f64, eta: f64) -> f64 {
    dielectric_fresnel(cosine_incident, eta).0
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

/// Cycles' `bsdf_microfacet_setup` threshold for a delta lobe: `alpha_x · alpha_y <= 2e-10`
/// (`roughness_is_almost_specular`, `kernel/closure/bsdf_microfacet.h`).
const GLASS_SINGULAR_ALPHA_SQUARED: f64 = 2.0e-10;

/// Cycles' GGX glass model. Thick glass uses one generalized-Schlick reflection/refraction closure; Thin Wall uses
/// the two-interface Fresnel weights and separate rough reflection/transmission closures (`ThinGlass`).
#[derive(Clone, Copy, Debug)]
struct GlassLobe {
    /// Thin-film coating in nanometers, with its IOR adjusted for a back-facing surface.
    thin_film_thickness: f64,
    thin_film_ior: f64,
    /// Cycles' `bsdf->ior`: the closure's IOR, inverted at a back face.
    ior: f64,
    /// GGX `alpha`, the roughness squared (Cycles' `alpha_x`).
    alpha: f64,
    /// Reflectance at normal incidence, tinted and saturated (`generalized_schlick_setup`'s `f0`).
    f0: DVec3,
    reflection_tint: DVec3,
    transmission_tint: DVec3,
    /// Multiscatter gain `1 + Fms · (1 - E) / E` per channel, from the view.
    gain: DVec3,
    /// Cycles' closure sample weight after its Fresnel albedo estimate and `MULTI_GGX` darkening.
    selection_weight: f64,
    singular: bool,
    straight: bool,
    thin: Option<ThinGlass>,
}

#[derive(Clone, Copy, Debug)]
struct ThinGlass {
    reflectance: DVec3,
    transmittance: DVec3,
    transmission_alpha: f64,
    reflection_gain: DVec3,
    transmission_gain: DVec3,
    transmission_enabled: bool,
}

/// The glass lobe at a shading point with normal `normal` seen along `view`, or `None` when its weight
/// `(1 - metallic) · transmission` is zero or the view is below the normal.
fn glass_lobe(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    reflective_caustics: bool,
    refractive_caustics: bool,
) -> Option<GlassLobe> {
    let metallic = params.metallic.clamp(0.0, 1.0);
    let transmission = params.transmission.clamp(0.0, 1.0);
    let cos_ni = normal.dot(view);
    if (1.0 - metallic) * transmission <= 0.0 || cos_ni <= 0.0 {
        return None;
    }
    let thin_film_thickness = params.thin_film_thickness;
    let roughness = params.roughness.clamp(0.0, 1.0);
    let alpha = roughness * roughness;
    let input_ior = params.ior.max(1.0e-5);
    let ior = if params.back_facing {
        1.0 / input_ior
    } else {
        input_ior
    };
    let thin_film_ior = params.thin_film_ior.max(1.0e-5);
    let thin_film_ior = if params.back_facing && thin_film_thickness > THICKNESS_CUTOFF {
        thin_film_ior / input_ior
    } else {
        thin_film_ior
    };
    // `generalized_schlick_setup` tints F0 of the closure's own IOR, which a back face does not invert.
    let f0 = (DVec3::from_array(params.specular_tint).max(DVec3::ZERO)
        * dielectric_f0(input_ior, 0.5))
    .clamp(DVec3::ZERO, DVec3::ONE);
    let reflection_tint = DVec3::from_array(params.glass_reflection_tint).max(DVec3::ZERO);
    let transmission_tint = DVec3::from_array(params.glass_transmission_tint).max(DVec3::ZERO);
    let singular = alpha * alpha <= GLASS_SINGULAR_ALPHA_SQUARED;
    let straight = (ior - 1.0).abs() < 1.0e-4;

    if params.thin_wall {
        let thin = thin_glass_setup(
            params,
            cos_ni,
            input_ior,
            roughness,
            alpha,
            reflective_caustics,
            refractive_caustics,
        );
        return Some(GlassLobe {
            thin_film_thickness,
            thin_film_ior: params.thin_film_ior.max(1.0e-5),
            ior: input_ior,
            alpha,
            f0,
            reflection_tint,
            transmission_tint,
            gain: thin.reflection_gain,
            selection_weight: 1.0,
            singular,
            straight: false,
            thin: Some(thin),
        });
    }

    // `bsdf_microfacet_setup_fresnel_generalized_schlick` first weights sampling by
    // `bsdf_microfacet_estimate_albedo`; MULTI_GGX then applies its preservation darkening.
    let (gain, energy, fss) = if params.glass_distribution == GlassDistribution::MultiGgx {
        let fss = if transmission_tint.max_element() <= 0.0 {
            // The reflective Fss: the dielectric's average single-scattering Fresnel, on the reflection tint.
            dielectric_fss(&SpecularLayer {
                eta: ior,
                f0_tinted: f0,
            }) * reflection_tint
        } else {
            transmission_tint
        };
        let inverse = ior < 1.0;
        let table_ior = if inverse { 1.0 / ior } else { ior };
        let z = ((table_ior - 1.0) / (table_ior + 1.0)).abs().sqrt();
        let (energy, energy_average) = if inverse {
            (
                ggx_glass_inverse_energy(roughness, cos_ni, z),
                ggx_glass_inverse_energy_average(roughness, z),
            )
        } else {
            (
                ggx_glass_energy(roughness, cos_ni, z),
                ggx_glass_energy_average(roughness, z),
            )
        };
        let energy = energy.max(1.0e-6);
        let missing = (1.0 - energy) / energy;
        (multiscatter_gain(fss, energy_average, missing), energy, fss)
    } else {
        (DVec3::ONE, 1.0, DVec3::ONE)
    };
    let mut selection_weight = glass_estimated_albedo(
        roughness,
        cos_ni,
        ior,
        thin_film_thickness,
        thin_film_ior,
        f0,
        reflection_tint,
        transmission_tint,
    );
    if params.glass_distribution == GlassDistribution::MultiGgx && fss != DVec3::ONE {
        // Cycles' darkening is `gain / energy_scale`, and `energy_scale` is `1 / E`.
        selection_weight *= (gain * energy).element_sum() / 3.0;
    }
    Some(GlassLobe {
        thin_film_thickness,
        thin_film_ior,
        ior,
        alpha,
        f0,
        reflection_tint,
        transmission_tint,
        gain,
        selection_weight,
        singular,
        straight,
        thin: None,
    })
}

fn glass_estimated_albedo(
    roughness: f64,
    cosine_incident: f64,
    ior: f64,
    thin_film_thickness: f64,
    thin_film_ior: f64,
    f0: DVec3,
    reflection_tint: DVec3,
    transmission_tint: DVec3,
) -> f64 {
    let fresnel = if thin_film_thickness > THICKNESS_CUTOFF {
        thin_film_dielectric_fresnel(cosine_incident, thin_film_thickness, thin_film_ior, ior, f0).0
    } else {
        let z = ((ior - 1.0) / (ior + 1.0)).abs().sqrt();
        let remap = ggx_gen_schlick_ior_s(roughness, cosine_incident, z);
        f0.lerp(DVec3::ONE, remap)
    };
    (fresnel * reflection_tint + (DVec3::ONE - fresnel) * transmission_tint).element_sum() / 3.0
}

fn thin_glass_setup(
    params: &BsdfParams,
    cosine_incident: f64,
    ior: f64,
    roughness: f64,
    alpha: f64,
    reflective_caustics: bool,
    refractive_caustics: bool,
) -> ThinGlass {
    let reflection_tint = DVec3::from_array(params.specular_tint).max(DVec3::ZERO);
    let f0 = reflection_tint * dielectric_f0(ior, 0.5);
    let has_film = params.thin_film_thickness > THICKNESS_CUTOFF;
    let film_ior = params.thin_film_ior.max(1.0e-5);
    let (fresnel, cos_theta_t) = if has_film {
        thin_film_dielectric_fresnel(
            cosine_incident,
            params.thin_film_thickness,
            film_ior,
            ior,
            f0,
        )
    } else {
        let (real, cos_theta_t) = dielectric_fresnel(cosine_incident, ior);
        (remapped_fresnel(f0, ior, real), cos_theta_t)
    };
    let r1 = if reflective_caustics {
        fresnel
    } else {
        DVec3::ZERO
    };
    let t1 = if refractive_caustics {
        DVec3::ONE - fresnel
    } else {
        DVec3::ZERO
    };
    let (r2, t2) = if has_film {
        let back_fresnel = thin_film_dielectric_fresnel(
            -cos_theta_t,
            params.thin_film_thickness,
            film_ior / ior,
            1.0 / ior,
            f0,
        )
        .0;
        (
            if reflective_caustics {
                back_fresnel
            } else {
                DVec3::ZERO
            },
            if refractive_caustics {
                DVec3::ONE - back_fresnel
            } else {
                DVec3::ZERO
            },
        )
    } else {
        (r1, t1)
    };
    let attenuation = if cos_theta_t == 0.0 {
        DVec3::ZERO
    } else {
        base_color(params).powf(-1.0 / cos_theta_t)
    };
    let back_reflection = r2 * attenuation;
    let numerator = attenuation * t1 * t2;
    let denominator = DVec3::ONE - back_reflection * back_reflection;
    let transmittance = DVec3::from_array(std::array::from_fn(|channel| {
        if denominator[channel] == 0.0 {
            0.0
        } else {
            numerator[channel] / denominator[channel]
        }
    }));
    let reflectance = r1 + transmittance * back_reflection;

    let energy = ggx_energy(roughness.clamp(MIN_ROUGHNESS, 1.0), cosine_incident).max(1.0e-6);
    let missing = (1.0 - energy) / energy;
    let reflection_gain = multiscatter_gain(
        reflection_tint,
        ggx_energy_average(roughness.clamp(MIN_ROUGHNESS, 1.0)),
        missing,
    );
    let roughness_factor = 3.4 * (ior - 1.0) * (ior - 0.5).powi(2) / (ior * ior * ior);
    let transmission_alpha = (alpha * roughness_factor.max(0.0).sqrt()).clamp(0.0, 1.0);
    let transmission_roughness = transmission_alpha.sqrt().clamp(MIN_ROUGHNESS, 1.0);
    let transmission_energy = ggx_energy(transmission_roughness, cosine_incident).max(1.0e-6);
    let transmission_gain = multiscatter_gain(
        DVec3::ONE,
        ggx_energy_average(transmission_roughness),
        (1.0 - transmission_energy) / transmission_energy,
    );
    ThinGlass {
        reflectance,
        transmittance,
        transmission_alpha,
        reflection_gain,
        transmission_gain,
        transmission_enabled: !params.thin_glass_transparent,
    }
}

impl GlassLobe {
    fn selection_weight(self) -> f64 {
        self.thin
            .map_or(self.selection_weight, ThinGlass::selection_weight)
    }
}

impl ThinGlass {
    fn selection_weight(self) -> f64 {
        (self.reflectance.element_sum()
            + if self.transmission_enabled {
                self.transmittance.element_sum()
            } else {
                0.0
            })
            / 3.0
    }

    fn reflection_probability(self) -> f64 {
        if !self.transmission_enabled {
            return 1.0;
        }
        let total = self.selection_weight();
        if total > 0.0 {
            self.reflectance.element_sum() / 3.0 / total
        } else {
            0.0
        }
    }
}

impl GlassLobe {
    /// Cycles' `generalized_schlick_fresnel` with the real dielectric remap: the reflectance and the
    /// transmittance at the cosine `cos_hi` between the half vector and the view, and the cosine of the
    /// refracted angle.
    fn fresnel(&self, cos_hi: f64) -> (DVec3, DVec3, f64) {
        let (fresnel, cos_ht) = if self.thin_film_thickness > THICKNESS_CUTOFF {
            thin_film_dielectric_fresnel(
                cos_hi,
                self.thin_film_thickness,
                self.thin_film_ior,
                self.ior,
                self.f0,
            )
        } else {
            let (real, cos_ht) = dielectric_fresnel(cos_hi, self.ior);
            (remapped_fresnel(self.f0, self.ior, real), cos_ht)
        };
        (
            fresnel * self.reflection_tint,
            (DVec3::ONE - fresnel) * self.transmission_tint,
            cos_ht,
        )
    }
}

/// Cycles' `bsdf_microfacet_eval` for the glass lobe. The value carries the cosine of the light direction,
/// as Cycles' eval does, so `value / |N·light|` is the BSDF. The density is in solid angle of `light`. The
/// reflection is taken above the normal and the transmission below it; both are zero for an invalid pair.
fn glass_eval(lobe: &GlassLobe, normal: DVec3, view: DVec3, light: DVec3) -> (DVec3, f64) {
    let cos_ni = normal.dot(view);
    let cos_no = normal.dot(light);
    if lobe.singular || cos_ni <= 0.0 || cos_no.abs() <= 1.0e-12 {
        return (DVec3::ZERO, 0.0);
    }
    let transmits = cos_no < 0.0;
    // Cycles' half vector: `-(ior · wo + wi)` for a transmission, `wi + wo` for a reflection.
    let unnormalised = if transmits {
        -(view + light * lobe.ior)
    } else {
        view + light
    };
    let length = unnormalised.length();
    if length <= 1.0e-12 {
        return (DVec3::ZERO, 0.0);
    }
    let half = unnormalised / length;
    let (reflectance, transmittance, _) = lobe.fresnel(half.dot(view));
    let total = (reflectance + transmittance).element_sum();
    if total <= 0.0 {
        return (DVec3::ZERO, 0.0);
    }
    let pdf_reflect = reflectance.element_sum() / total;
    let alpha_squared = lobe.alpha * lobe.alpha;
    let distribution = ggx_distribution(alpha_squared, normal.dot(half));
    let lambda_i = ggx_lambda(alpha_squared, cos_ni);
    let lambda_o = ggx_lambda(alpha_squared, cos_no.abs());
    let masking = 1.0 / (1.0 + lambda_i + lambda_o);
    if transmits {
        let ior_over_length = lobe.ior / length;
        let common = distribution / cos_ni
            * ior_over_length
            * ior_over_length
            * (half.dot(view) * half.dot(light)).abs();
        (
            transmittance * (common * masking) * lobe.gain,
            (1.0 - pdf_reflect) * common / (1.0 + lambda_i),
        )
    } else {
        let common = 0.25 * distribution / cos_ni;
        (
            reflectance * (common * masking) * lobe.gain,
            pdf_reflect * common / (1.0 + lambda_i),
        )
    }
}

fn thin_glass_density(
    lobe: &GlassLobe,
    normal: DVec3,
    view: DVec3,
    light: DVec3,
    branch: f64,
) -> LobeDensity {
    let Some(thin) = lobe.thin else {
        return LobeDensity {
            weight: 0.0,
            roughness_squared: 0.0,
        };
    };
    let reflection_probability = thin.reflection_probability();
    if normal.dot(light) > 0.0 {
        if lobe.alpha * lobe.alpha <= GLASS_SINGULAR_ALPHA_SQUARED {
            return LobeDensity {
                weight: 0.0,
                roughness_squared: 0.0,
            };
        }
        LobeDensity {
            weight: branch
                * reflection_probability
                * visible_half_vector_pdf(
                    normal,
                    isotropic_microfacet(lobe.alpha.sqrt()),
                    view,
                    light,
                ),
            roughness_squared: lobe.alpha * lobe.alpha,
        }
    } else if normal.dot(light) < 0.0 {
        if !thin.transmission_enabled {
            return LobeDensity {
                weight: 0.0,
                roughness_squared: 0.0,
            };
        }
        let alpha = thin.transmission_alpha;
        if alpha * alpha <= GLASS_SINGULAR_ALPHA_SQUARED {
            return LobeDensity {
                weight: 0.0,
                roughness_squared: 0.0,
            };
        }
        let virtual_view = view - normal * (2.0 * normal.dot(view));
        LobeDensity {
            weight: branch
                * (1.0 - reflection_probability)
                * visible_half_vector_pdf(
                    -normal,
                    isotropic_microfacet(alpha.sqrt()),
                    virtual_view,
                    light,
                ),
            roughness_squared: alpha * alpha,
        }
    } else {
        LobeDensity {
            weight: 0.0,
            roughness_squared: 0.0,
        }
    }
}

fn thin_microfacet_eval(
    normal: DVec3,
    view: DVec3,
    light: DVec3,
    alpha: f64,
    color: DVec3,
    gain: DVec3,
) -> (DVec3, f64) {
    let cos_ni = normal.dot(view);
    let cos_no = normal.dot(light);
    if alpha * alpha <= GLASS_SINGULAR_ALPHA_SQUARED || cos_ni <= 0.0 || cos_no <= 0.0 {
        return (DVec3::ZERO, 0.0);
    }
    let half = (view + light).normalize_or_zero();
    let n_dot_h = normal.dot(half);
    if n_dot_h <= 0.0 {
        return (DVec3::ZERO, 0.0);
    }
    let alpha_squared = alpha * alpha;
    let distribution = ggx_distribution(alpha_squared, n_dot_h);
    let lambda_i = ggx_lambda(alpha_squared, cos_ni);
    let lambda_o = ggx_lambda(alpha_squared, cos_no);
    let common = 0.25 * distribution / cos_ni;
    let value = color * (common / (1.0 + lambda_i + lambda_o)) * gain;
    let pdf = visible_half_vector_pdf(normal, isotropic_microfacet(alpha.sqrt()), view, light);
    (value, pdf)
}

fn thin_glass_eval(lobe: &GlassLobe, normal: DVec3, view: DVec3, light: DVec3) -> (DVec3, f64) {
    let Some(thin) = lobe.thin else {
        return (DVec3::ZERO, 0.0);
    };
    if normal.dot(light) > 0.0 {
        thin_microfacet_eval(
            normal,
            view,
            light,
            lobe.alpha,
            thin.reflectance,
            thin.reflection_gain,
        )
    } else if normal.dot(light) < 0.0 && thin.transmission_enabled {
        let virtual_view = view - normal * (2.0 * normal.dot(view));
        thin_microfacet_eval(
            -normal,
            virtual_view,
            light,
            thin.transmission_alpha,
            thin.transmittance,
            thin.transmission_gain,
        )
    } else {
        (DVec3::ZERO, 0.0)
    }
}

/// The thin-glass transmission share Cycles turns into a transparent closure for non-camera rays when it is singular.
pub(crate) fn thin_glass_shadow_transparency(params: &BsdfParams, layers: &Layers) -> DVec3 {
    let Some(lobe) = layers.glass else {
        return DVec3::ZERO;
    };
    let Some(thin) = lobe.thin else {
        return DVec3::ZERO;
    };
    if thin.transmission_alpha * thin.transmission_alpha > GLASS_SINGULAR_ALPHA_SQUARED {
        return DVec3::ZERO;
    }
    let amount = (1.0 - params.metallic.clamp(0.0, 1.0)) * params.transmission.clamp(0.0, 1.0);
    thin.transmittance * (amount * params.reflectance.clamp(0.0, 1.0)) * layers.lower
}

/// The glass lobe along `light`, times its Principled transmission share, reflectance and the layers above it: reflection
/// above the surface and transmission below. A delta lobe is zero here, since `sample_glass` draws it.
pub(crate) fn glass_coefficient(params: &BsdfParams, layers: &Layers, light: DVec3) -> DVec3 {
    let Some(lobe) = layers.glass.as_ref() else {
        return DVec3::ZERO;
    };
    let cosine = layers.specular.dot(light).abs();
    if cosine <= 0.0 {
        return DVec3::ZERO;
    }
    let value = if lobe.thin.is_some() {
        thin_glass_eval(lobe, layers.specular, layers.view, light).0
    } else {
        glass_eval(lobe, layers.specular, layers.view, light).0
    };
    value * ((1.0 - params.metallic.clamp(0.0, 1.0)) * params.transmission.clamp(0.0, 1.0)) / cosine
        * (params.reflectance.clamp(0.0, 1.0) * layers.lower)
}

/// Cycles' `1e6` for a singular draw's pdf (`bsdf_microfacet_sample`: `*pdf *= 1e6f` when `m_singular`).
pub(crate) const SINGULAR_MIS_WEIGHT: f64 = 1.0e6;

/// A glass draw: thick glass samples its combined Fresnel closure; Thin Wall samples one of its two closures.
pub(crate) enum GlassSample {
    /// A delta reflection or transmission. `weight` is the selected spectral closure share over the branch probability,
    /// times its energy gain, reflectance and layers. `mis_weight` is `SINGULAR_MIS_WEIGHT × branch × share`.
    Singular {
        direction: DVec3,
        weight: DVec3,
        mis_weight: f64,
    },
    /// A rough draw, which the caller weights by the lobe mixture's density.
    Rough { direction: DVec3 },
    /// A draw on the wrong side of the normal or the geometric normal: Cycles' `LABEL_NONE`.
    Absorbed,
}

/// Draws the thick glass closure by sampling a visible normal and Fresnel reflection/refraction. Thin Wall selects its
/// reflection or flipped-normal mirror-transmission closure before sampling that closure's GGX normal.
pub(crate) fn sample_glass(
    params: &BsdfParams,
    layers: &Layers,
    geometric: DVec3,
    branch: f64,
    rands: [f64; 3],
) -> GlassSample {
    let Some(lobe) = layers.glass.as_ref() else {
        return GlassSample::Absorbed;
    };
    let (normal, view) = (layers.specular, layers.view);
    if lobe.thin.is_some() {
        return sample_thin_glass(params, layers, lobe, geometric, branch, rands);
    }
    let half = if lobe.singular {
        normal
    } else {
        let (tangent, bitangent) = make_orthonormals(normal);
        let local_view = DVec3::new(view.dot(tangent), view.dot(bitangent), normal.dot(view));
        let local_half = sample_visible_normal(local_view, lobe.alpha, rands[0], rands[1]);
        tangent * local_half.x + bitangent * local_half.y + normal * local_half.z
    };
    let cos_hi = half.dot(view);
    let (reflectance, transmittance, cos_ht) = lobe.fresnel(cos_hi);
    let total = (reflectance + transmittance).element_sum();
    if total <= 0.0 {
        return GlassSample::Absorbed;
    }
    let pdf_reflect = reflectance.element_sum() / total;
    let refracts = rands[2] >= pdf_reflect;
    // Cycles' `refract_angle` for a refraction, and the mirror of the view for a reflection.
    let direction = if refracts {
        (half * (cos_hi / lobe.ior + cos_ht) - view / lobe.ior).normalize_or_zero()
    } else {
        (half * (2.0 * cos_hi) - view).normalize_or_zero()
    };
    if (geometric.dot(direction) < 0.0) != refracts || (normal.dot(direction) < 0.0) != refracts {
        return GlassSample::Absorbed;
    }
    // Cycles' `m_singular` also covers a refraction at an IOR of 1 (`|eta - 1| < 1e-4`).
    if !(lobe.singular || (refracts && lobe.straight)) {
        return GlassSample::Rough { direction };
    }
    let (value, share) = if refracts {
        (transmittance, 1.0 - pdf_reflect)
    } else {
        (reflectance, pdf_reflect)
    };
    let glass_weight =
        (1.0 - params.metallic.clamp(0.0, 1.0)) * params.transmission.clamp(0.0, 1.0);
    let scale = glass_weight * params.reflectance.clamp(0.0, 1.0) * layers.lower;
    GlassSample::Singular {
        direction,
        weight: value * lobe.gain * scale / (branch * share),
        mis_weight: SINGULAR_MIS_WEIGHT * branch * share,
    }
}

fn sample_thin_glass(
    params: &BsdfParams,
    layers: &Layers,
    lobe: &GlassLobe,
    geometric: DVec3,
    branch: f64,
    rands: [f64; 3],
) -> GlassSample {
    let Some(thin) = lobe.thin else {
        return GlassSample::Absorbed;
    };
    let normal = layers.specular;
    let view = layers.view;
    let reflection_probability = thin.reflection_probability();
    let reflects = !thin.transmission_enabled || rands[2] < reflection_probability;
    let (sample_normal, sample_view, alpha, color, gain, share) = if reflects {
        (
            normal,
            view,
            lobe.alpha,
            thin.reflectance,
            thin.reflection_gain,
            reflection_probability,
        )
    } else {
        (
            -normal,
            view - normal * (2.0 * normal.dot(view)),
            thin.transmission_alpha,
            thin.transmittance,
            thin.transmission_gain,
            1.0 - reflection_probability,
        )
    };
    if share <= 0.0 {
        return GlassSample::Absorbed;
    }
    let half = if alpha * alpha <= GLASS_SINGULAR_ALPHA_SQUARED {
        sample_normal
    } else {
        let (tangent, bitangent) = make_orthonormals(sample_normal);
        let local_view = DVec3::new(
            sample_view.dot(tangent),
            sample_view.dot(bitangent),
            sample_normal.dot(sample_view),
        );
        let local_half = sample_visible_normal(local_view, alpha, rands[0], rands[1]);
        tangent * local_half.x + bitangent * local_half.y + sample_normal * local_half.z
    };
    let incoming = -sample_view;
    let direction = (incoming - half * (2.0 * half.dot(incoming))).normalize_or_zero();
    if if reflects {
        geometric.dot(direction) <= 0.0
    } else {
        geometric.dot(direction) >= 0.0
    } {
        return GlassSample::Absorbed;
    }
    if alpha * alpha > GLASS_SINGULAR_ALPHA_SQUARED {
        return GlassSample::Rough { direction };
    }
    let amount = (1.0 - params.metallic.clamp(0.0, 1.0)) * params.transmission.clamp(0.0, 1.0);
    GlassSample::Singular {
        direction,
        weight: color * gain * (amount * params.reflectance.clamp(0.0, 1.0)) * layers.lower
            / (branch * share),
        mis_weight: SINGULAR_MIS_WEIGHT * branch * share,
    }
}

/// Cycles' `microfacet_ggx_sample_vndf` (Heitz 2018): the isotropic half vector visible from local view `wi`.
pub(super) fn sample_visible_normal(wi: DVec3, alpha: f64, u: f64, v: f64) -> DVec3 {
    sample_visible_normal_anisotropic(wi, alpha, alpha, u, v)
}

/// Cycles' anisotropic GGX VNDF half vector visible from local view `wi` (`z` up).
fn sample_visible_normal_anisotropic(
    wi: DVec3,
    alpha_x: f64,
    alpha_y: f64,
    u: f64,
    v: f64,
) -> DVec3 {
    let stretched = DVec3::new(alpha_x * wi.x, alpha_y * wi.y, wi.z).normalize_or_zero();
    let length_squared = stretched.x * stretched.x + stretched.y * stretched.y;
    let (t1, t2) = if length_squared > 1.0e-7 {
        let t1 = DVec3::new(-stretched.y, stretched.x, 0.0) / length_squared.sqrt();
        (t1, stretched.cross(t1))
    } else {
        (DVec3::X, DVec3::Y)
    };
    let (x, disk_y) = concentric_disk(u, v);
    let s = f64::midpoint(1.0, stretched.z);
    let y = (1.0 - s) * (1.0 - x * x).max(0.0).sqrt() + s * disk_y;
    let hemisphere = DVec3::new(x, y, (1.0 - x * x - y * y).max(0.0).sqrt());
    let half = t1 * hemisphere.x + t2 * hemisphere.y + stretched * hemisphere.z;
    DVec3::new(alpha_x * half.x, alpha_y * half.y, half.z.max(0.0)).normalize_or_zero()
}

pub(super) fn sample_specular_ggx_half(
    microfacet: GgxMicrofacet,
    normal: DVec3,
    view: DVec3,
    u: f64,
    v: f64,
) -> DVec3 {
    let (tangent, bitangent) = if microfacet.anisotropic {
        (microfacet.tangent, microfacet.bitangent)
    } else {
        make_orthonormals(normal)
    };
    let local_view = DVec3::new(tangent.dot(view), bitangent.dot(view), normal.dot(view));
    let local_half =
        sample_visible_normal_anisotropic(local_view, microfacet.alpha_x, microfacet.alpha_y, u, v);
    (tangent * local_half.x + bitangent * local_half.y + normal * local_half.z).normalize_or_zero()
}

/// The result of sampling one of the non-Principled standalone surface closures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum StandaloneSample {
    Continuous {
        direction: DVec3,
        ray: RayKind,
    },
    Delta {
        direction: DVec3,
        weight: DVec3,
        ray: RayKind,
        branch_pdf: f64,
    },
    Subsurface {
        weight: DVec3,
        method: SubsurfaceMethod,
    },
    Absorbed,
}

/// The standalone closure's BSDF coefficient, excluding the cosine of `light`.
pub(crate) fn standalone_eval(params: &BsdfParams, layers: &Layers, light: DVec3) -> DVec3 {
    match params.model {
        BsdfModel::Sheen(SheenDistribution::Microfiber) => {
            standalone_sheen_lobe(params, layers).value(light) * standalone_color(params)
        }
        BsdfModel::Sheen(SheenDistribution::Ashikhmin) => {
            let Some((cosine_weighted, _)) = standalone_velvet_eval(params, layers, light) else {
                return DVec3::ZERO;
            };
            cosine_weighted * standalone_color(params) / layers.normal.dot(light)
        }
        BsdfModel::Subsurface => {
            let (_, diffuse_weight, _, _, _) = standalone_subsurface_weights(params);
            if layers.specular.dot(light) > 0.0 {
                diffuse_weight / PI
            } else {
                DVec3::ZERO
            }
        }
        BsdfModel::Translucent => {
            if layers.specular.dot(light) < 0.0 {
                standalone_color(params) / PI
            } else {
                DVec3::ZERO
            }
        }
        BsdfModel::Refraction(
            distribution @ (MicrofacetDistribution::Beckmann | MicrofacetDistribution::Ggx),
        ) => {
            let microfacet = standalone_refraction_microfacet(params, layers, distribution);
            let (cosine_weighted, _) =
                microfacet_refraction_eval(microfacet, layers.view, light, standalone_eta(params));
            let cosine = layers.specular.dot(light).abs();
            if cosine > 0.0 {
                cosine_weighted * standalone_color(params) / cosine
            } else {
                DVec3::ZERO
            }
        }
        BsdfModel::Toon(toon) => {
            let (cosine_weighted, _) = standalone_toon_eval(toon, layers, light);
            let cosine = layers.normal.dot(light).abs();
            if cosine > 0.0 {
                cosine_weighted * standalone_color(params) / cosine
            } else {
                DVec3::ZERO
            }
        }
        BsdfModel::Metallic(metallic) => {
            let microfacet = standalone_metallic_microfacet(params, layers, metallic.distribution);
            let (cosine_weighted, _) = microfacet_reflection_eval(
                microfacet,
                layers.view,
                light,
                standalone_metallic_fresnel(
                    params,
                    metallic,
                    layers.view.dot((layers.view + light).normalize_or_zero()),
                ),
            );
            let cosine = layers.specular.dot(light).abs();
            if cosine > 0.0 {
                cosine_weighted * standalone_metallic_gain(params, metallic, layers) / cosine
            } else {
                DVec3::ZERO
            }
        }
        BsdfModel::Refraction(MicrofacetDistribution::MultiGgx) => {
            unreachable!("the evaluator rejects Multi-GGX on the Refraction node")
        }
        BsdfModel::Principled
        | BsdfModel::Holdout
        | BsdfModel::Hair(_)
        | BsdfModel::PrincipledHair(_)
        | BsdfModel::RayPortal(_) => {
            unreachable!("standalone BSDF kernels do not handle this closure model")
        }
    }
}

/// The standalone closure's continuous sample density in solid angle.
pub(crate) fn standalone_density(params: &BsdfParams, layers: &Layers, light: DVec3) -> f64 {
    match params.model {
        BsdfModel::Sheen(SheenDistribution::Microfiber) => {
            standalone_sheen_lobe(params, layers).density(light)
        }
        BsdfModel::Sheen(SheenDistribution::Ashikhmin) => {
            standalone_velvet_eval(params, layers, light).map_or(0.0, |(_, pdf)| pdf)
        }
        BsdfModel::Subsurface => {
            let (_, _, _, diffuse_probability, _) = standalone_subsurface_weights(params);
            diffuse_probability * layers.specular.dot(light).max(0.0) / PI
        }
        BsdfModel::Translucent => (-layers.specular.dot(light)).max(0.0) / PI,
        BsdfModel::Refraction(
            distribution @ (MicrofacetDistribution::Beckmann | MicrofacetDistribution::Ggx),
        ) => {
            let microfacet = standalone_refraction_microfacet(params, layers, distribution);
            microfacet_refraction_eval(microfacet, layers.view, light, standalone_eta(params)).1
        }
        BsdfModel::Toon(toon) => standalone_toon_eval(toon, layers, light).1,
        BsdfModel::Metallic(metallic) => {
            let microfacet = standalone_metallic_microfacet(params, layers, metallic.distribution);
            let fresnel = standalone_metallic_fresnel(
                params,
                metallic,
                layers.view.dot((layers.view + light).normalize_or_zero()),
            );
            microfacet_reflection_eval(microfacet, layers.view, light, fresnel).1
        }
        BsdfModel::Refraction(MicrofacetDistribution::MultiGgx) => {
            unreachable!("the evaluator rejects Multi-GGX on the Refraction node")
        }
        BsdfModel::Principled
        | BsdfModel::Holdout
        | BsdfModel::Hair(_)
        | BsdfModel::PrincipledHair(_)
        | BsdfModel::RayPortal(_) => {
            unreachable!("standalone BSDF kernels do not handle this closure model")
        }
    }
}

/// Cycles' post-setup closure `sample_weight`, excluding the outer mix/closure weight and reflective share.
/// `path.rs` multiplies those scalar weights once when it selects among closures.
pub(crate) fn standalone_selection_weight(params: &BsdfParams, layers: &Layers) -> f64 {
    match params.model {
        BsdfModel::Sheen(SheenDistribution::Microfiber) => {
            standalone_color_sample_weight(params) * standalone_sheen_lobe(params, layers).weight.x
        }
        BsdfModel::Sheen(SheenDistribution::Ashikhmin)
        | BsdfModel::Translucent
        | BsdfModel::Refraction(MicrofacetDistribution::Beckmann | MicrofacetDistribution::Ggx)
        | BsdfModel::Toon(_) => standalone_color_sample_weight(params),
        BsdfModel::Subsurface => {
            let color_weight = standalone_color_sample_weight(params);
            if color_weight == 0.0 {
                return 0.0;
            }
            standalone_subsurface_weights(params).4
        }
        BsdfModel::Metallic(metallic) => {
            standalone_metallic_selection_weight(params, layers, metallic)
        }
        BsdfModel::Refraction(MicrofacetDistribution::MultiGgx) => {
            unreachable!("the evaluator rejects Multi-GGX on the Refraction node")
        }
        BsdfModel::Principled
        | BsdfModel::Holdout
        | BsdfModel::Hair(_)
        | BsdfModel::PrincipledHair(_)
        | BsdfModel::RayPortal(_) => {
            unreachable!("standalone BSDF kernels do not handle this closure model")
        }
    }
}

fn standalone_color_sample_weight(params: &BsdfParams) -> f64 {
    (standalone_color(params).element_sum() / 3.0).abs()
}
fn standalone_metallic_selection_weight(
    params: &BsdfParams,
    layers: &Layers,
    metallic: MetallicParameters,
) -> f64 {
    let microfacet = standalone_metallic_setup_microfacet(params, layers, metallic.distribution);
    let cosine_i = microfacet.normal.dot(layers.view);
    let roughness = (microfacet.alpha_x * microfacet.alpha_y).sqrt().sqrt();
    let estimated_albedo = match metallic.fresnel {
        MetallicFresnel::F82 if params.thin_film_thickness.max(1.0e-5) <= THICKNESS_CUTOFF => {
            let f0 = standalone_color(params).clamp(DVec3::ZERO, DVec3::ONE);
            f0.lerp(DVec3::ONE, ggx_gen_schlick_s_f82(roughness, cosine_i))
        }
        _ => standalone_metallic_fresnel(params, metallic, cosine_i),
    };
    let mut sample_weight = estimated_albedo.element_sum() / 3.0;
    if metallic.distribution == MicrofacetDistribution::MultiGgx {
        let energy = ggx_energy(roughness, cosine_i);
        let darkening = standalone_metallic_gain(params, metallic, layers) * energy;
        sample_weight *= darkening.element_sum() / 3.0;
    }
    sample_weight
}

/// Samples the standalone closure using the exact Cycles 5.2.2 distribution and geometric-normal rejection.
pub(crate) fn standalone_sample(
    params: &BsdfParams,
    layers: &Layers,
    geometric: DVec3,
    random: [f64; 3],
) -> StandaloneSample {
    match params.model {
        BsdfModel::Sheen(SheenDistribution::Microfiber) => {
            let lobe = standalone_sheen_lobe(params, layers);
            let direction = lobe.direction(random[0], random[1]);
            if lobe.weight.max_element() <= 0.0 || geometric.dot(direction) <= 0.0 {
                StandaloneSample::Absorbed
            } else {
                StandaloneSample::Continuous {
                    direction,
                    ray: RayKind::Diffuse,
                }
            }
        }
        BsdfModel::Sheen(SheenDistribution::Ashikhmin) => {
            let direction = sample_uniform_hemisphere(layers.normal, random[0], random[1]);
            let cosine_i = layers.normal.dot(layers.view);
            let half = (layers.view + direction).normalize_or_zero();
            let cosine_nh = layers.normal.dot(half);
            let cosine_hi = layers.view.dot(half).abs();
            // Cycles' sample rejects against geometric Ng, then checks the incoming half-vector; unlike eval, it does
            // not reject a sampled direction solely for falling below the shading normal.
            if cosine_i <= 1.0e-5
                || geometric.dot(direction) <= 0.0
                || !(cosine_nh.abs() < 1.0 - 1.0e-5 && cosine_hi > 1.0e-5)
            {
                StandaloneSample::Absorbed
            } else {
                StandaloneSample::Continuous {
                    direction,
                    ray: RayKind::Diffuse,
                }
            }
        }
        BsdfModel::Subsurface => {
            let (subsurface_weight, _, subsurface_probability, _, _) =
                standalone_subsurface_weights(params);
            if random[0] < subsurface_probability {
                StandaloneSample::Subsurface {
                    weight: subsurface_weight / subsurface_probability,
                    method: params.subsurface_method,
                }
            } else {
                let direction = sample_cosine_hemisphere(layers.specular, random[1], random[2]);
                if geometric.dot(direction) <= 0.0 {
                    StandaloneSample::Absorbed
                } else {
                    StandaloneSample::Continuous {
                        direction,
                        ray: RayKind::Diffuse,
                    }
                }
            }
        }
        BsdfModel::Translucent => {
            let direction = sample_cosine_hemisphere(-layers.specular, random[0], random[1]);
            if geometric.dot(direction) >= 0.0 {
                StandaloneSample::Absorbed
            } else {
                StandaloneSample::Continuous {
                    direction,
                    ray: RayKind::Transmission,
                }
            }
        }
        BsdfModel::Refraction(
            distribution @ (MicrofacetDistribution::Beckmann | MicrofacetDistribution::Ggx),
        ) => standalone_refraction_sample(params, layers, geometric, distribution, random),
        BsdfModel::Toon(toon) => standalone_toon_sample(toon, layers, geometric, random),
        BsdfModel::Metallic(metallic) => {
            standalone_metallic_sample(params, layers, geometric, metallic, random)
        }
        BsdfModel::Refraction(MicrofacetDistribution::MultiGgx) => {
            unreachable!("the evaluator rejects Multi-GGX on the Refraction node")
        }
        BsdfModel::Principled
        | BsdfModel::Holdout
        | BsdfModel::Hair(_)
        | BsdfModel::PrincipledHair(_)
        | BsdfModel::RayPortal(_) => {
            unreachable!("standalone BSDF kernels do not handle this closure model")
        }
    }
}

fn standalone_color(params: &BsdfParams) -> DVec3 {
    DVec3::new(
        params.base_color[0],
        params.base_color[1],
        params.base_color[2],
    )
}

fn standalone_sheen_lobe(params: &BsdfParams, layers: &Layers) -> SheenLobe {
    let normal = layers.normal;
    let cosine = normal.dot(layers.view);
    let off = SheenLobe {
        normal,
        tangent: DVec3::ZERO,
        bitangent: DVec3::ZERO,
        transform_a: 0.0,
        transform_b: 0.0,
        weight: DVec3::ZERO,
    };
    if cosine <= 0.0 {
        return off;
    }
    let roughness = saturate(params.sheen_roughness).clamp(1.0e-3, 1.0);
    let ltc = sheen_ltc(roughness, cosine);
    if ltc.transform_a.abs() < 1.0e-5 || ltc.albedo < 1.0e-5 {
        return off;
    }
    let (tangent, bitangent) = sheen_frame(normal, layers.view);
    SheenLobe {
        normal,
        tangent,
        bitangent,
        transform_a: ltc.transform_a,
        transform_b: ltc.transform_b,
        weight: DVec3::splat(ltc.albedo),
    }
}

fn standalone_velvet_eval(
    params: &BsdfParams,
    layers: &Layers,
    light: DVec3,
) -> Option<(f64, f64)> {
    let normal = layers.normal;
    let cosine_i = normal.dot(layers.view);
    let cosine_o = normal.dot(light);
    if cosine_i <= 0.0 || cosine_o <= 0.0 {
        return None;
    }
    let half = (layers.view + light).normalize_or_zero();
    let cosine_nh = normal.dot(half);
    let cosine_hi = layers.view.dot(half).abs();
    if !(cosine_nh.abs() < 1.0 - 1.0e-5 && cosine_hi > 1.0e-5) {
        return None;
    }
    let sigma = params.sheen_roughness.clamp(0.0, 1.0).max(0.01);
    let inverse_sigma_squared = 1.0 / (sigma * sigma);
    let cosine_nh_div_hi = (cosine_nh / cosine_hi).max(1.0e-5);
    let fac1 = 2.0 * (cosine_nh_div_hi * cosine_i).abs();
    let fac2 = 2.0 * (cosine_nh_div_hi * cosine_o).abs();
    let sine_nh_squared = 1.0 - cosine_nh * cosine_nh;
    let sine_nh_fourth = sine_nh_squared * sine_nh_squared;
    let cotangent_squared = cosine_nh * cosine_nh / sine_nh_squared;
    let distribution =
        (-cotangent_squared * inverse_sigma_squared).exp() * inverse_sigma_squared * (1.0 / PI)
            / sine_nh_fourth;
    let masking = 1.0_f64.min(fac1).min(fac2);
    let cosine_weighted = 0.25 * (distribution * masking) / cosine_i;
    Some((cosine_weighted, 0.5 / PI))
}

fn sample_uniform_hemisphere(normal: DVec3, u: f64, v: f64) -> DVec3 {
    let (mut x, mut y) = concentric_disk(u, v);
    let z = 1.0 - x * x - y * y;
    let scale = (z + 1.0).max(0.0).sqrt();
    x *= scale;
    y *= scale;
    let (tangent, bitangent) = make_orthonormals(normal);
    tangent * x + bitangent * y + normal * z
}

fn sample_cosine_hemisphere(normal: DVec3, u: f64, v: f64) -> DVec3 {
    let (x, y) = concentric_disk(u, v);
    let cosine = (1.0 - x * x - y * y).max(0.0).sqrt();
    let (tangent, bitangent) = make_orthonormals(normal);
    tangent * x + bitangent * y + normal * cosine
}

fn standalone_subsurface_weights(params: &BsdfParams) -> (DVec3, DVec3, f64, f64, f64) {
    let mask = channel_mask(raw_radius(params));
    let active = DVec3::from_array(mask.map(|enabled| {
        if enabled && params.subsurface_weight > 0.0 {
            1.0
        } else {
            0.0
        }
    }));
    let color = standalone_color(params);
    let subsurface = color * active;
    let diffuse = color * (DVec3::ONE - active);
    let channel_count = mask.iter().filter(|&&enabled| enabled).count() as f64;
    let subsurface_sample_weight = (subsurface.element_sum() / 3.0).abs() * channel_count;
    let diffuse_sample_weight = (diffuse.element_sum() / 3.0).abs();
    let total = subsurface_sample_weight + diffuse_sample_weight;
    if total > 0.0 {
        (
            subsurface,
            diffuse,
            subsurface_sample_weight / total,
            diffuse_sample_weight / total,
            total,
        )
    } else {
        (subsurface, diffuse, 0.0, 1.0, 0.0)
    }
}

#[derive(Clone, Copy)]
struct StandaloneMicrofacet {
    distribution: MicrofacetDistribution,
    normal: DVec3,
    tangent: DVec3,
    bitangent: DVec3,
    alpha_x: f64,
    alpha_y: f64,
    anisotropic: bool,
}

impl StandaloneMicrofacet {
    fn isotropic(distribution: MicrofacetDistribution, normal: DVec3, alpha: f64) -> Self {
        let (tangent, bitangent) = make_orthonormals(normal);
        Self {
            distribution,
            normal,
            tangent,
            bitangent,
            alpha_x: alpha,
            alpha_y: alpha,
            anisotropic: false,
        }
    }

    fn local(self, vector: DVec3) -> DVec3 {
        DVec3::new(
            self.tangent.dot(vector),
            self.bitangent.dot(vector),
            self.normal.dot(vector),
        )
    }
    fn roughness_squared(self) -> f64 {
        self.alpha_x * self.alpha_y
    }
}

fn standalone_refraction_microfacet(
    params: &BsdfParams,
    layers: &Layers,
    distribution: MicrofacetDistribution,
) -> StandaloneMicrofacet {
    let alpha = params
        .roughness
        .powi(2)
        .clamp(0.0, 1.0)
        .max(layers.glossy_blur_alpha);
    match distribution {
        MicrofacetDistribution::Beckmann | MicrofacetDistribution::Ggx => {
            StandaloneMicrofacet::isotropic(distribution, layers.specular, alpha)
        }
        MicrofacetDistribution::MultiGgx => {
            unreachable!("the evaluator rejects Multi-GGX on the Refraction node")
        }
    }
}

/// Cycles' setup alpha of a metallic closure, before the glossy blur. Its Fresnel albedo and multiscatter gain are
/// fixed at setup (`sample_weight`, `energy_scale`), so they read this rather than [`standalone_metallic_microfacet`].
fn standalone_metallic_setup_microfacet(
    params: &BsdfParams,
    layers: &Layers,
    distribution: MicrofacetDistribution,
) -> StandaloneMicrofacet {
    let alpha = params.roughness.clamp(0.0, 1.0).powi(2);
    let mut alpha_x = alpha;
    let mut alpha_y = alpha;
    let mut tangent_frame = None;
    let anisotropy = params.anisotropy.clamp(0.0, 1.0);
    if anisotropy > 0.0
        && let Some(tangent) = params.anisotropic_tangent
    {
        let tangent = rotate_around_axis(
            tangent,
            layers.normal,
            params.anisotropic_rotation * 2.0 * PI,
        );
        let aspect = (1.0 - 0.9 * anisotropy).sqrt();
        alpha_x /= aspect;
        alpha_y *= aspect;
        tangent_frame = Some(tangent);
    }
    alpha_x = saturate(alpha_x);
    alpha_y = saturate(alpha_y);
    let Some(tangent) = tangent_frame.filter(|_| !exact_f64_eq(alpha_x, alpha_y)) else {
        return StandaloneMicrofacet::isotropic(
            distribution,
            layers.specular,
            alpha_x.max(alpha_y),
        );
    };
    let bitangent = layers.specular.cross(tangent).normalize_or_zero();
    if bitangent.length_squared() <= f64::EPSILON {
        return StandaloneMicrofacet::isotropic(
            distribution,
            layers.specular,
            alpha_x.max(alpha_y),
        );
    }
    StandaloneMicrofacet {
        distribution,
        normal: layers.specular,
        tangent: bitangent.cross(layers.specular),
        bitangent,
        alpha_x,
        alpha_y,
        anisotropic: true,
    }
}

/// The metallic microfacet after Cycles' glossy blur: `bsdf_microfacet_blur` raises both axes to `max(blur, alpha)`.
fn standalone_metallic_microfacet(
    params: &BsdfParams,
    layers: &Layers,
    distribution: MicrofacetDistribution,
) -> StandaloneMicrofacet {
    let mut microfacet = standalone_metallic_setup_microfacet(params, layers, distribution);
    microfacet.alpha_x = microfacet.alpha_x.max(layers.glossy_blur_alpha);
    microfacet.alpha_y = microfacet.alpha_y.max(layers.glossy_blur_alpha);
    microfacet
}

fn standalone_eta(params: &BsdfParams) -> f64 {
    let ior = params.ior.max(1.0e-5);
    if params.back_facing { 1.0 / ior } else { ior }
}

fn standalone_toon_parameters(toon: ToonParameters) -> (f64, f64, f64) {
    let size = toon.size.clamp(1.0e-5, 1.0) * FRAC_PI_2;
    let smooth = toon.smooth.clamp(0.0, 1.0) * FRAC_PI_2;
    (size, smooth, (size + smooth).min(FRAC_PI_2))
}

fn toon_intensity(size: f64, smooth: f64, angle: f64) -> f64 {
    if angle < size {
        1.0
    } else if angle < size + smooth && smooth != 0.0 {
        1.0 - (angle - size) / smooth
    } else {
        0.0
    }
}

fn sample_uniform_cone(normal: DVec3, one_minus_cos_angle: f64, u: f64, v: f64) -> DVec3 {
    if one_minus_cos_angle > 0.0 {
        let (mut x, mut y) = concentric_disk(u, v);
        let radius_squared = x * x + y * y;
        let cosine = 1.0 - radius_squared * one_minus_cos_angle;
        let scale = (one_minus_cos_angle * (2.0 - one_minus_cos_angle * radius_squared))
            .max(0.0)
            .sqrt();
        x *= scale;
        y *= scale;
        let (tangent, bitangent) = make_orthonormals(normal);
        tangent * x + bitangent * y + normal * cosine
    } else {
        normal
    }
}

fn standalone_toon_eval(toon: ToonParameters, layers: &Layers, light: DVec3) -> (DVec3, f64) {
    let (size, smooth, sample_angle) = standalone_toon_parameters(toon);
    let sample_pdf = 0.5 / (PI * (1.0 - sample_angle.cos()));
    let normal = layers.normal;
    let cosine_o = normal.dot(light);
    if cosine_o <= 0.0 {
        return (DVec3::ZERO, 0.0);
    }
    match toon.component {
        ToonComponent::Diffuse => {
            let angle = cosine_o.clamp(0.0, 1.0).acos();
            if angle < sample_angle {
                (
                    DVec3::splat(sample_pdf * toon_intensity(size, smooth, angle)),
                    sample_pdf,
                )
            } else {
                (DVec3::ZERO, 0.0)
            }
        }
        ToonComponent::Glossy => {
            let cosine_i = normal.dot(layers.view);
            if cosine_i <= 0.0 {
                return (DVec3::ZERO, 0.0);
            }
            let reflection = 2.0 * cosine_i * normal - layers.view;
            let cosine_ro = reflection.dot(light);
            let angle = cosine_ro.max(0.0).clamp(0.0, 1.0).acos();
            if angle < sample_angle {
                (
                    DVec3::splat(sample_pdf * toon_intensity(size, smooth, angle)),
                    sample_pdf,
                )
            } else {
                (DVec3::ZERO, 0.0)
            }
        }
    }
}

fn standalone_toon_sample(
    toon: ToonParameters,
    layers: &Layers,
    geometric: DVec3,
    random: [f64; 3],
) -> StandaloneSample {
    let (_, _, sample_angle) = standalone_toon_parameters(toon);
    let normal = layers.normal;
    let (center, ray) = match toon.component {
        ToonComponent::Diffuse => (normal, RayKind::Diffuse),
        ToonComponent::Glossy => {
            let cosine_i = normal.dot(layers.view);
            if cosine_i <= 0.0 {
                return StandaloneSample::Absorbed;
            }
            (2.0 * cosine_i * normal - layers.view, RayKind::Glossy)
        }
    };
    let direction = sample_uniform_cone(center, 1.0 - sample_angle.cos(), random[0], random[1]);
    if geometric.dot(direction) <= 0.0
        || (toon.component == ToonComponent::Glossy && normal.dot(direction) <= 0.0)
    {
        return StandaloneSample::Absorbed;
    }
    StandaloneSample::Continuous { direction, ray }
}

fn standalone_microfacet_distribution(microfacet: StandaloneMicrofacet, half: DVec3) -> f64 {
    let half = microfacet.local(half);
    let cosine_squared = (half.z * half.z).min(1.0);
    let one_minus_cosine_squared = 1.0 - cosine_squared;
    let alpha_squared = microfacet.alpha_x * microfacet.alpha_y;
    match (microfacet.distribution, microfacet.anisotropic) {
        (MicrofacetDistribution::Beckmann, false) => {
            1.0 / ((one_minus_cosine_squared / (cosine_squared * alpha_squared)).exp()
                * PI
                * alpha_squared
                * cosine_squared.powi(2))
        }
        (MicrofacetDistribution::Ggx | MicrofacetDistribution::MultiGgx, false) => {
            alpha_squared
                / (PI * (one_minus_cosine_squared + alpha_squared * cosine_squared).powi(2))
        }
        (MicrofacetDistribution::Beckmann, true) => {
            let scaled = DVec3::new(
                half.x / microfacet.alpha_x,
                half.y / microfacet.alpha_y,
                half.z,
            );
            let cosine_squared = scaled.z * scaled.z;
            (-(scaled.x * scaled.x + scaled.y * scaled.y) / cosine_squared).exp()
                / (PI * alpha_squared * cosine_squared.powi(2))
        }
        (MicrofacetDistribution::Ggx | MicrofacetDistribution::MultiGgx, true) => {
            let scaled = DVec3::new(
                half.x / microfacet.alpha_x,
                half.y / microfacet.alpha_y,
                half.z,
            );
            1.0 / (PI * alpha_squared * scaled.length_squared().powi(2))
        }
    }
}

fn beckmann_lambda_from_alpha_tan(alpha_tan_squared: f64) -> f64 {
    if alpha_tan_squared < 0.39 {
        return 0.0;
    }
    let a = alpha_tan_squared.sqrt().recip();
    ((0.396 * a - 1.259) * a + 1.0) / ((2.181 * a + 3.535) * a)
}

fn standalone_microfacet_lambda(microfacet: StandaloneMicrofacet, direction: DVec3) -> f64 {
    let local = microfacet.local(direction);
    let alpha_tan_squared = if microfacet.anisotropic {
        ((microfacet.alpha_x * local.x).powi(2) + (microfacet.alpha_y * local.y).powi(2))
            / local.z.powi(2)
    } else {
        microfacet.alpha_x * microfacet.alpha_y * (1.0 / local.z.powi(2) - 1.0).max(0.0)
    };
    match microfacet.distribution {
        MicrofacetDistribution::Beckmann => beckmann_lambda_from_alpha_tan(alpha_tan_squared),
        MicrofacetDistribution::Ggx | MicrofacetDistribution::MultiGgx => {
            0.5 * ((1.0 + alpha_tan_squared).sqrt() - 1.0)
        }
    }
}

fn microfacet_reflection_eval(
    microfacet: StandaloneMicrofacet,
    view: DVec3,
    light: DVec3,
    fresnel: DVec3,
) -> (DVec3, f64) {
    let cosine_i = microfacet.normal.dot(view);
    let cosine_o = microfacet.normal.dot(light);
    if cosine_i <= 0.0
        || cosine_o <= 0.0
        || microfacet.alpha_x * microfacet.alpha_y <= GLASS_SINGULAR_ALPHA_SQUARED
        || fresnel.max_element() <= 0.0
    {
        return (DVec3::ZERO, 0.0);
    }
    let half = (view + light).normalize_or_zero();
    if half == DVec3::ZERO {
        return (DVec3::ZERO, 0.0);
    }
    let distribution = standalone_microfacet_distribution(microfacet, half);
    let lambda_i = standalone_microfacet_lambda(microfacet, view);
    let lambda_o = standalone_microfacet_lambda(microfacet, light);
    let common = distribution / cosine_i * 0.25;
    (
        fresnel * (common / (1.0 + lambda_i + lambda_o)),
        common / (1.0 + lambda_i),
    )
}

fn microfacet_refraction_eval(
    microfacet: StandaloneMicrofacet,
    view: DVec3,
    light: DVec3,
    eta: f64,
) -> (DVec3, f64) {
    let cosine_i = microfacet.normal.dot(view);
    let cosine_o = microfacet.normal.dot(light);
    if cosine_i <= 0.0
        || cosine_o >= 0.0
        || microfacet.alpha_x * microfacet.alpha_y <= GLASS_SINGULAR_ALPHA_SQUARED
    {
        return (DVec3::ZERO, 0.0);
    }
    let half_unscaled = -(eta * light + view);
    let half_length = half_unscaled.length();
    if half_length == 0.0 {
        return (DVec3::ZERO, 0.0);
    }
    let half = half_unscaled / half_length;
    let cosine_hi = half.dot(view);
    let (fresnel, _) = dielectric_fresnel(cosine_hi, eta);
    if fresnel >= 1.0 {
        return (DVec3::ZERO, 0.0);
    }
    let distribution = standalone_microfacet_distribution(microfacet, half);
    let lambda_i = standalone_microfacet_lambda(microfacet, view);
    let lambda_o = standalone_microfacet_lambda(microfacet, light);
    let common =
        distribution / cosine_i * (eta / half_length).powi(2) * (cosine_hi * half.dot(light)).abs();
    (
        DVec3::splat(common / (1.0 + lambda_i + lambda_o)),
        common / (1.0 + lambda_i),
    )
}

fn sample_standalone_half(microfacet: StandaloneMicrofacet, view: DVec3, u: f64, v: f64) -> DVec3 {
    let local_view = microfacet.local(view);
    let local_half = match microfacet.distribution {
        MicrofacetDistribution::Ggx | MicrofacetDistribution::MultiGgx => {
            sample_visible_normal_anisotropic(
                local_view,
                microfacet.alpha_x,
                microfacet.alpha_y,
                u,
                v,
            )
        }
        MicrofacetDistribution::Beckmann => {
            sample_visible_beckmann_normal(local_view, microfacet.alpha_x, microfacet.alpha_y, u, v)
        }
    };
    (microfacet.tangent * local_half.x
        + microfacet.bitangent * local_half.y
        + microfacet.normal * local_half.z)
        .normalize_or_zero()
}

fn sample_visible_beckmann_normal(
    view: DVec3,
    alpha_x: f64,
    alpha_y: f64,
    u: f64,
    v: f64,
) -> DVec3 {
    let stretched = DVec3::new(alpha_x * view.x, alpha_y * view.y, view.z).normalize_or_zero();
    let (mut slope_x, mut slope_y, cosine_phi, sine_phi);
    if stretched.z >= 0.99999 {
        let radius = (-u.ln()).sqrt();
        let phi = 2.0 * PI * v;
        slope_x = radius * phi.cos();
        slope_y = radius * phi.sin();
        cosine_phi = 1.0;
        sine_phi = 0.0;
    } else {
        let cosine_theta = stretched.z;
        let sine_theta = (1.0 - cosine_theta * cosine_theta).sqrt();
        let tangent_theta = sine_theta / cosine_theta;
        let cotangent_theta = 1.0 / tangent_theta;
        let erf_a = cycles_fast_erf(cotangent_theta);
        let exp_a2 = (-cotangent_theta * cotangent_theta).exp();
        let inverse_sine_theta = 1.0 / sine_theta;
        cosine_phi = stretched.x * inverse_sine_theta;
        sine_phi = stretched.y * inverse_sine_theta;
        let k = tangent_theta * (1.0 / PI.sqrt());
        let y_approx = u * (1.0 + erf_a + k * (1.0 - erf_a * erf_a));
        let y_exact = u * (1.0 + erf_a + k * exp_a2);
        let b = if k > 0.0 {
            (0.5 - (k * (k - y_approx + 1.0) + 0.25).sqrt()) / k
        } else {
            y_approx - 1.0
        };
        let mut inverse_erf = cycles_fast_erfinv(b);
        let mut begin_x = -1.0;
        let mut begin_y = -y_exact;
        let mut end_x = erf_a;
        let mut current_x = b;
        let mut current_y = 1.0 + b + k * (-inverse_erf * inverse_erf).exp() - y_exact;
        let mut iteration = 0;
        while current_y.abs() > 1.0e-6 && iteration < 3 {
            if begin_y.signum() == current_y.signum() {
                begin_x = current_x;
                begin_y = current_y;
            } else {
                end_x = current_x;
            }
            let newton_x = current_x - current_y / (1.0 - inverse_erf * tangent_theta);
            current_x = if newton_x >= begin_x && newton_x <= end_x {
                newton_x
            } else {
                f64::midpoint(begin_x, end_x)
            };
            inverse_erf = cycles_fast_erfinv(current_x);
            current_y = 1.0 + current_x + k * (-inverse_erf * inverse_erf).exp() - y_exact;
            iteration += 1;
        }
        slope_x = inverse_erf;
        slope_y = cycles_fast_erfinv(2.0 * v - 1.0);
    }
    let rotated_x = cosine_phi * slope_x - sine_phi * slope_y;
    let rotated_y = sine_phi * slope_x + cosine_phi * slope_y;
    slope_x = alpha_x * rotated_x;
    slope_y = alpha_y * rotated_y;
    DVec3::new(-slope_x, -slope_y, 1.0).normalize_or_zero()
}

fn cycles_fast_erf(value: f64) -> f64 {
    let a = value.abs();
    if a >= 12.3 {
        return value.signum();
    }
    let b = 1.0 - (1.0 - a);
    let mut r = 0.000_043_063_8;
    r = r * b + 0.000_276_567_2;
    r = r * b + 0.000_152_014_3;
    r = r * b + 0.009_270_527_2;
    r = r * b + 0.042_282_012_3;
    r = r * b + 0.070_523_078_4;
    r = r * b + 1.0;
    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;
    let r16 = r8 * r8;
    value.signum() * (1.0 - 1.0 / r16)
}

fn cycles_fast_erfinv(value: f64) -> f64 {
    let x = value.abs().min(0.999_999_94);
    let mut w = -((1.0 - x) * (1.0 + x)).ln();
    let p = if w < 5.0 {
        w -= 2.5;
        let mut p = 2.810_226_36e-8;
        p = p * w + 3.432_739_39e-7;
        p = p * w - 3.523_387_7e-6;
        p = p * w - 4.391_506_54e-6;
        p = p * w + 0.000_218_580_87;
        p = p * w - 0.001_253_725_03;
        p = p * w - 0.004_177_681_64;
        p = p * w + 0.246_640_727;
        p = p * w + 1.501_409_41;
        p
    } else {
        w = w.sqrt() - 3.0;
        let mut p = -0.000_200_214_257;
        p = p * w + 0.000_100_950_558;
        p = p * w + 0.001_349_343_22;
        p = p * w - 0.003_673_428_44;
        p = p * w + 0.005_739_507_73;
        p = p * w - 0.007_622_461_3;
        p = p * w + 0.009_438_870_47;
        p = p * w + 1.001_674_06;
        p = p * w + 2.832_976_82;
        p
    };
    p * value
}

/// Cycles' `table_ggx_gen_schlick_s` z slices 7 and 8; F82 setup queries the table at z = 0.5.
#[rustfmt::skip]
const GGX_GEN_SCHLICK_S_F82_Z_SLICES: [f32; 512] = [
    0.999_429_f32, 0.902_696_f32, 0.741_916_f32, 0.583_382_f32, 0.447_240_f32, 0.338_334_f32, 0.254_889_f32, 0.192_674_f32,
    0.147_061_f32, 0.113_921_f32, 0.089_921_f32, 0.072_527_f32, 0.059_860_f32, 0.050_578_f32, 0.043_717_f32, 0.038_600_f32,
    0.674_189_f32, 0.669_511_f32, 0.625_563_f32, 0.529_453_f32, 0.419_763_f32, 0.320_911_f32, 0.240_949_f32, 0.180_307_f32,
    0.135_826_f32, 0.103_679_f32, 0.080_539_f32, 0.063_802_f32, 0.051_596_f32, 0.042_573_f32, 0.035_809_f32, 0.030_663_f32,
    0.441_437_f32, 0.440_114_f32, 0.426_338_f32, 0.386_542_f32, 0.324_527_f32, 0.257_868_f32, 0.198_748_f32, 0.151_123_f32,
    0.114_754_f32, 0.087_755_f32, 0.067_948_f32, 0.053_435_f32, 0.042_727_f32, 0.034_737_f32, 0.028_688_f32, 0.024_027_f32,
    0.279_401_f32, 0.278_969_f32, 0.274_104_f32, 0.258_187_f32, 0.228_435_f32, 0.190_125_f32, 0.151_635_f32, 0.118_175_f32,
    0.091_267_f32, 0.070_519_f32, 0.054_862_f32, 0.043_135_f32, 0.034_332_f32, 0.027_673_f32, 0.022_576_f32, 0.018_620_f32,
    0.169_939_f32, 0.169_831_f32, 0.168_349_f32, 0.162_641_f32, 0.150_062_f32, 0.130_926_f32, 0.108_791_f32, 0.087_496_f32,
    0.069_170_f32, 0.054_328_f32, 0.042_704_f32, 0.033_746_f32, 0.026_864_f32, 0.021_574_f32, 0.017_471_f32, 0.014_263_f32,
    0.098_574_f32, 0.098_583_f32, 0.098_421_f32, 0.097_069_f32, 0.092_861_f32, 0.084_820_f32, 0.073_750_f32, 0.061_627_f32,
    0.050_172_f32, 0.040_276_f32, 0.032_150_f32, 0.025_641_f32, 0.020_509_f32, 0.016_470_f32, 0.013_296_f32, 0.010_785_f32,
    0.053_987_f32, 0.054_031_f32, 0.054_313_f32, 0.054_634_f32, 0.054_119_f32, 0.051_758_f32, 0.047_264_f32, 0.041_276_f32,
    0.034_831_f32, 0.028_738_f32, 0.023_406_f32, 0.018_925_f32, 0.015_263_f32, 0.012_305_f32, 0.009_933_f32, 0.008_035_f32,
    0.027_542_f32, 0.027_584_f32, 0.027_937_f32, 0.028_710_f32, 0.029_522_f32, 0.029_655_f32, 0.028_565_f32, 0.026_230_f32,
    0.023_095_f32, 0.019_697_f32, 0.016_444_f32, 0.013_538_f32, 0.011_052_f32, 0.008_977_f32, 0.007_273_f32, 0.005_886_f32,
    0.012_841_f32, 0.012_873_f32, 0.013_157_f32, 0.013_875_f32, 0.014_906_f32, 0.015_837_f32, 0.016_196_f32, 0.015_743_f32,
    0.014_556_f32, 0.012_915_f32, 0.011_113_f32, 0.009_353_f32, 0.007_760_f32, 0.006_372_f32, 0.005_199_f32, 0.004_224_f32,
    0.005_322_f32, 0.005_341_f32, 0.005_526_f32, 0.006_032_f32, 0.006_854_f32, 0.007_793_f32, 0.008_546_f32, 0.008_862_f32,
    0.008_665_f32, 0.008_048_f32, 0.007_176_f32, 0.006_207_f32, 0.005_254_f32, 0.004_378_f32, 0.003_609_f32, 0.002_953_f32,
    0.001_878_f32, 0.001_888_f32, 0.001_989_f32, 0.002_284_f32, 0.002_805_f32, 0.003_480_f32, 0.004_147_f32, 0.004_632_f32,
    0.004_827_f32, 0.004_722_f32, 0.004_390_f32, 0.003_922_f32, 0.003_402_f32, 0.002_889_f32, 0.002_415_f32, 0.001_998_f32,
    0.000_525_f32, 0.000_529_f32, 0.000_575_f32, 0.000_718_f32, 0.000_992_f32, 0.001_382_f32, 0.001_821_f32, 0.002_215_f32,
    0.002_478_f32, 0.002_572_f32, 0.002_507_f32, 0.002_328_f32, 0.002_080_f32, 0.001_809_f32, 0.001_540_f32, 0.001_293_f32,
    0.000_101_f32, 0.000_103_f32, 0.000_120_f32, 0.000_176_f32, 0.000_294_f32, 0.000_478_f32, 0.000_709_f32, 0.000_947_f32,
    0.001_146_f32, 0.001_270_f32, 0.001_307_f32, 0.001_269_f32, 0.001_176_f32, 0.001_053_f32, 0.000_918_f32, 0.000_785_f32,
    0.000_010_f32, 0.000_010_f32, 0.000_015_f32, 0.000_032_f32, 0.000_072_f32, 0.000_140_f32, 0.000_236_f32, 0.000_348_f32,
    0.000_457_f32, 0.000_543_f32, 0.000_595_f32, 0.000_609_f32, 0.000_590_f32, 0.000_548_f32, 0.000_493_f32, 0.000_433_f32,
    0.000_000_f32, 0.000_000_f32, 0.000_001_f32, 0.000_005_f32, 0.000_014_f32, 0.000_033_f32, 0.000_061_f32, 0.000_099_f32,
    0.000_141_f32, 0.000_181_f32, 0.000_213_f32, 0.000_233_f32, 0.000_239_f32, 0.000_234_f32, 0.000_220_f32, 0.000_200_f32,
    0.000_000_f32, 0.000_000_f32, 0.000_000_f32, 0.000_001_f32, 0.000_002_f32, 0.000_004_f32, 0.000_008_f32, 0.000_013_f32,
    0.000_021_f32, 0.000_029_f32, 0.000_039_f32, 0.000_047_f32, 0.000_054_f32, 0.000_058_f32, 0.000_059_f32, 0.000_058_f32,
    0.999_563_f32, 0.920_276_f32, 0.782_410_f32, 0.639_695_f32, 0.510_912_f32, 0.402_654_f32, 0.315_576_f32, 0.247_554_f32,
    0.195_444_f32, 0.156_013_f32, 0.126_382_f32, 0.104_180_f32, 0.087_530_f32, 0.075_010_f32, 0.065_547_f32, 0.058_351_f32,
    0.739_454_f32, 0.735_078_f32, 0.693_262_f32, 0.600_892_f32, 0.493_225_f32, 0.391_952_f32, 0.305_836_f32, 0.237_299_f32,
    0.184_748_f32, 0.145_215_f32, 0.115_727_f32, 0.093_726_f32, 0.077_246_f32, 0.064_787_f32, 0.055_272_f32, 0.047_918_f32,
    0.534_691_f32, 0.533_264_f32, 0.518_413_f32, 0.475_399_f32, 0.407_724_f32, 0.333_383_f32, 0.265_184_f32, 0.208_036_f32,
    0.162_649_f32, 0.127_711_f32, 0.101_242_f32, 0.081_296_f32, 0.066_227_f32, 0.054_754_f32, 0.045_920_f32, 0.039_017_f32,
    0.376_720_f32, 0.376_139_f32, 0.369_819_f32, 0.349_699_f32, 0.312_606_f32, 0.264_761_f32, 0.215_928_f32, 0.172_392_f32,
    0.136_340_f32, 0.107_724_f32, 0.085_543_f32, 0.068_532_f32, 0.055_502_f32, 0.045_471_f32, 0.037_676_f32, 0.031_544_f32,
    0.257_451_f32, 0.257_225_f32, 0.254_584_f32, 0.245_439_f32, 0.226_607_f32, 0.198_944_f32, 0.167_278_f32, 0.136_636_f32,
    0.109_873_f32, 0.087_796_f32, 0.070_176_f32, 0.056_355_f32, 0.045_568_f32, 0.037_153_f32, 0.030_540_f32, 0.025_304_f32,
    0.169_668_f32, 0.169_606_f32, 0.168_708_f32, 0.165_090_f32, 0.156_544_f32, 0.142_219_f32, 0.123_694_f32, 0.103_936_f32,
    0.085_396_f32, 0.069_327_f32, 0.056_024_f32, 0.045_266_f32, 0.036_691_f32, 0.029_872_f32, 0.024_452_f32, 0.020_115_f32,
    0.107_007_f32, 0.107_017_f32, 0.106_936_f32, 0.106_056_f32, 0.103_058_f32, 0.096_797_f32, 0.087_297_f32, 0.075_812_f32,
    0.063_990_f32, 0.053_032_f32, 0.043_509_f32, 0.035_512_f32, 0.028_952_f32, 0.023_619_f32, 0.019_308_f32, 0.015_823_f32,
    0.063_917_f32, 0.063_953_f32, 0.064_206_f32, 0.064_558_f32, 0.064_324_f32, 0.062_536_f32, 0.058_644_f32, 0.052_890_f32,
    0.046_122_f32, 0.039_222_f32, 0.032_808_f32, 0.027_160_f32, 0.022_350_f32, 0.018_332_f32, 0.015_016_f32, 0.012_295_f32,
    0.035_637_f32, 0.035_676_f32, 0.036_012_f32, 0.036_785_f32, 0.037_694_f32, 0.038_081_f32, 0.037_300_f32, 0.035_128_f32,
    0.031_831_f32, 0.027_939_f32, 0.023_951_f32, 0.020_190_f32, 0.016_834_f32, 0.013_927_f32, 0.011_468_f32, 0.009_413_f32,
    0.018_156_f32, 0.018_189_f32, 0.018_486_f32, 0.019_262_f32, 0.020_434_f32, 0.021_610_f32, 0.022_271_f32, 0.022_053_f32,
    0.020_904_f32, 0.019_056_f32, 0.016_833_f32, 0.014_521_f32, 0.012_315_f32, 0.010_315_f32, 0.008_565_f32, 0.007_068_f32,
    0.008_177_f32, 0.008_200_f32, 0.008_414_f32, 0.009_014_f32, 0.010_019_f32, 0.011_236_f32, 0.012_322_f32, 0.012_950_f32,
    0.012_948_f32, 0.012_343_f32, 0.011_306_f32, 0.010_034_f32, 0.008_694_f32, 0.007_403_f32, 0.006_222_f32, 0.005_181_f32,
    0.003_080_f32, 0.003_094_f32, 0.003_223_f32, 0.003_605_f32, 0.004_295_f32, 0.005_223_f32, 0.006_201_f32, 0.007_005_f32,
    0.007_458_f32, 0.007_498_f32, 0.007_171_f32, 0.006_588_f32, 0.005_866_f32, 0.005_101_f32, 0.004_358_f32, 0.003_676_f32,
    0.000_875_f32, 0.000_881_f32, 0.000_945_f32, 0.001_144_f32, 0.001_530_f32, 0.002_095_f32, 0.002_763_f32, 0.003_411_f32,
    0.003_912_f32, 0.004_187_f32, 0.004_216_f32, 0.004_040_f32, 0.003_720_f32, 0.003_324_f32, 0.002_902_f32, 0.002_491_f32,
    0.000_148_f32, 0.000_151_f32, 0.000_174_f32, 0.000_254_f32, 0.000_421_f32, 0.000_691_f32, 0.001_046_f32, 0.001_438_f32,
    0.001_800_f32, 0.002_072_f32, 0.002_219_f32, 0.002_239_f32, 0.002_153_f32, 0.001_993_f32, 0.001_792_f32, 0.001_575_f32,
    0.000_007_f32, 0.000_008_f32, 0.000_013_f32, 0.000_034_f32, 0.000_085_f32, 0.000_176_f32, 0.000_312_f32, 0.000_483_f32,
    0.000_668_f32, 0.000_838_f32, 0.000_968_f32, 0.001_044_f32, 0.001_063_f32, 0.001_034_f32, 0.000_970_f32, 0.000_885_f32,
    0.000_000_f32, 0.000_000_f32, 0.000_001_f32, 0.000_004_f32, 0.000_011_f32, 0.000_027_f32, 0.000_054_f32, 0.000_094_f32,
    0.000_146_f32, 0.000_205_f32, 0.000_266_f32, 0.000_319_f32, 0.000_360_f32, 0.000_383_f32, 0.000_389_f32, 0.000_381_f32,
];
fn ggx_gen_schlick_s_f82(roughness: f64, cosine_i: f64) -> f64 {
    let slice = |z_index: usize| lookup_ggx_gen_schlick_s_slice(z_index, roughness, cosine_i);
    0.5 * slice(0) + 0.5 * slice(1)
}

fn lookup_ggx_gen_schlick_s_slice(z_index: usize, roughness: f64, cosine_i: f64) -> f64 {
    let x = roughness.clamp(0.0, 1.0) * 15.0;
    let x_index = (x.floor() as usize).min(15);
    let x_next = (x_index + 1).min(15);
    let x_fraction = x - x_index as f64;
    let y = cosine_i.clamp(0.0, 1.0) * 15.0;
    let y_index = (y.floor() as usize).min(15);
    let y_next = (y_index + 1).min(15);
    let y_fraction = y - y_index as f64;
    let row = |index: usize| {
        let offset = z_index * 256 + index * 16;
        let d0 = f64::from(GGX_GEN_SCHLICK_S_F82_Z_SLICES[offset + x_index]);
        if x_fraction == 0.0 {
            return d0;
        }
        let d1 = f64::from(GGX_GEN_SCHLICK_S_F82_Z_SLICES[offset + x_next]);
        (1.0 - x_fraction) * d0 + x_fraction * d1
    };
    let d0 = row(y_index);
    if y_fraction == 0.0 {
        return d0;
    }
    let d1 = row(y_next);
    (1.0 - y_fraction) * d0 + y_fraction * d1
}

fn standalone_metallic_fresnel(
    params: &BsdfParams,
    metallic: MetallicParameters,
    cosine: f64,
) -> DVec3 {
    let thickness = params.thin_film_thickness.max(1.0e-5);
    let film_ior = params.thin_film_ior.max(1.0e-5);
    match metallic.fresnel {
        MetallicFresnel::F82 => {
            let f0 = standalone_color(params).clamp(DVec3::ZERO, DVec3::ONE);
            let tint = DVec3::from_array(metallic.edge_tint).clamp(DVec3::ZERO, DVec3::ONE);
            let b = f82_tint_b(f0, tint);
            if thickness > THICKNESS_CUTOFF {
                thin_film_f82_fresnel(cosine, thickness, film_ior, f0, b)
            } else {
                f82_fresnel(cosine, f0, b)
            }
        }
        MetallicFresnel::PhysicalConductor => {
            let ior = DVec3::from_array(metallic.conductor_ior).max(DVec3::ZERO);
            let extinction = DVec3::from_array(metallic.extinction).max(DVec3::ZERO);
            if thickness > THICKNESS_CUTOFF {
                physical_conductor_thin_film_fresnel(cosine, thickness, film_ior, ior, extinction)
            } else {
                physical_conductor_fresnel(cosine, ior, extinction)
            }
        }
    }
}

fn physical_conductor_fresnel(cosine: f64, ior: DVec3, extinction: DVec3) -> DVec3 {
    DVec3::from_array(std::array::from_fn(|channel| {
        let n = ior[channel];
        let k = extinction[channel];
        let eta_squared = n * n;
        let extinction_squared = k * k;
        let two_n_k = 2.0 * n * k;
        let t1 = eta_squared - extinction_squared - (1.0 - cosine * cosine);
        let t2 = (t1 * t1 + two_n_k * two_n_k).sqrt();
        let u_squared = f64::midpoint(t2, t1).max(0.0);
        let v_squared = (0.5 * (t2 - t1)).max(0.0);
        let u = u_squared.sqrt();
        let v = v_squared.sqrt();
        let rs = safe_ratio(
            (cosine - u).powi(2) + v_squared,
            (cosine + u).powi(2) + v_squared,
        );
        let t3 = (eta_squared - extinction_squared) * cosine;
        let t4 = two_n_k * cosine;
        let rp = safe_ratio(
            (t3 - u).powi(2) + (t4 - v).powi(2),
            (t3 + u).powi(2) + (t4 + v).powi(2),
        );
        f64::midpoint(rs, rp)
    }))
}

#[derive(Clone, Copy)]
struct MetallicFilmComplex {
    re: f64,
    im: f64,
}

impl MetallicFilmComplex {
    fn multiply(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

fn physical_conductor_thin_film_fresnel(
    cosine: f64,
    thickness: f64,
    film_ior: f64,
    substrate_ior: DVec3,
    extinction: DVec3,
) -> DVec3 {
    DVec3::from_array(std::array::from_fn(|channel| {
        physical_conductor_thin_film_channel(
            channel,
            cosine,
            thickness,
            film_ior,
            substrate_ior[channel],
            extinction[channel],
        )
    }))
}

fn physical_conductor_thin_film_channel(
    channel: usize,
    cosine_incident: f64,
    thickness: f64,
    film_ior: f64,
    substrate_n: f64,
    substrate_k: f64,
) -> f64 {
    let film_ior = if thickness < 1.0 {
        let x = saturate(thickness);
        let smooth = x * x * (3.0 - 2.0 * x);
        1.0 + (film_ior - 1.0) * smooth
    } else {
        film_ior
    };
    let Some((r12_s, r12_p, cos_theta_2, phase12_s, phase12_p)) =
        thin_film_dielectric_interface(cosine_incident, film_ior)
    else {
        return 1.0;
    };
    if exact_f64_eq(r12_s, 1.0) && exact_f64_eq(r12_p, 1.0) {
        return 1.0;
    }
    let (r23_s, r23_p, phase23_s, phase23_p) =
        physical_conductor_film_interface(-cos_theta_2, film_ior, substrate_n, substrate_k);
    let optical_path_difference = -2.0 * film_ior * thickness * cos_theta_2;
    let phase_s = phase23_s.multiply(MetallicFilmComplex {
        re: -phase12_s,
        im: 0.0,
    });
    let phase_p = phase23_p.multiply(MetallicFilmComplex {
        re: -phase12_p,
        im: 0.0,
    });
    let reflectance_s =
        metallic_film_airy_summation(channel, r12_s, r23_s, optical_path_difference, phase_s);
    let reflectance_p =
        metallic_film_airy_summation(channel, r12_p, r23_p, optical_path_difference, phase_p);
    saturate(f64::midpoint(reflectance_s, reflectance_p))
}

fn thin_film_dielectric_interface(
    cosine_incident: f64,
    eta: f64,
) -> Option<(f64, f64, f64, f64, f64)> {
    let cosine_incident = cosine_incident.abs();
    let eta_cosine_transmitted_squared = eta * eta - (1.0 - cosine_incident * cosine_incident);
    if eta_cosine_transmitted_squared <= 0.0 {
        return None;
    }
    let cosine_transmitted = -eta_cosine_transmitted_squared.sqrt() / eta;
    let r_s =
        (cosine_incident + eta * cosine_transmitted) / (cosine_incident - eta * cosine_transmitted);
    let r_p =
        (cosine_transmitted + eta * cosine_incident) / (eta * cosine_incident - cosine_transmitted);
    Some((
        r_s * r_s,
        r_p * r_p,
        cosine_transmitted,
        if r_s >= 0.0 { 1.0 } else { -1.0 },
        if r_p >= 0.0 { 1.0 } else { -1.0 },
    ))
}

fn physical_conductor_film_interface(
    cosine_incident: f64,
    ambient_ior: f64,
    substrate_n: f64,
    substrate_k: f64,
) -> (f64, f64, MetallicFilmComplex, MetallicFilmComplex) {
    let ambient_ior_squared = ambient_ior * ambient_ior;
    let substrate_n_squared = substrate_n * substrate_n;
    let substrate_k_squared = substrate_k * substrate_k;
    let two_n_k = 2.0 * substrate_n * substrate_k;
    let t1 = substrate_n_squared
        - substrate_k_squared
        - ambient_ior_squared * (1.0 - cosine_incident * cosine_incident);
    let t2 = (t1 * t1 + two_n_k * two_n_k).sqrt();
    let u_squared = f64::midpoint(t2, t1).max(0.0);
    let v_squared = (0.5 * (t2 - t1)).max(0.0);
    let u = u_squared.sqrt();
    let v = v_squared.sqrt();
    let ambient_cosine = ambient_ior * cosine_incident;
    let reflectance_s = safe_ratio(
        (ambient_cosine - u).powi(2) + v_squared,
        (ambient_cosine + u).powi(2) + v_squared,
    );
    let t3 = (substrate_n_squared - substrate_k_squared) * cosine_incident;
    let t4 = two_n_k * cosine_incident;
    let reflectance_p = safe_ratio(
        (t3 - ambient_ior * u).powi(2) + (t4 - ambient_ior * v).powi(2),
        (t3 + ambient_ior * u).powi(2) + (t4 + ambient_ior * v).powi(2),
    );
    let phase_s = metallic_film_unit_complex(
        -u_squared - v_squared + ambient_cosine.powi(2),
        -2.0 * ambient_cosine * v,
    );
    let phase_p = metallic_film_unit_complex(
        ((substrate_n_squared + substrate_k_squared) * cosine_incident).powi(2)
            - ambient_ior_squared * (u_squared + v_squared),
        2.0 * ambient_ior
            * cosine_incident
            * (two_n_k * u - (substrate_n_squared - substrate_k_squared) * v),
    );
    (reflectance_s, reflectance_p, phase_s, phase_p)
}

fn metallic_film_unit_complex(re: f64, im: f64) -> MetallicFilmComplex {
    let magnitude = (re * re + im * im).sqrt();
    if magnitude == 0.0 {
        MetallicFilmComplex { re: 1.0, im: 0.0 }
    } else {
        MetallicFilmComplex {
            re: re / magnitude,
            im: im / magnitude,
        }
    }
}

fn metallic_film_airy_summation(
    channel: usize,
    reflectance_12: f64,
    reflectance_23: f64,
    optical_path_difference: f64,
    phasor: MetallicFilmComplex,
) -> f64 {
    let transmission_12_squared = 1.0 - reflectance_12;
    let round_trip_reflectance = reflectance_12 * reflectance_23;
    let round_trip_root = round_trip_reflectance.sqrt();
    let single_reflection = transmission_12_squared * transmission_12_squared * reflectance_23
        / (1.0 - round_trip_reflectance);
    let mut reflectance = single_reflection + reflectance_12;
    let mut coefficient = single_reflection - transmission_12_squared;
    let mut accumulator = phasor;
    for order in 1..4 {
        coefficient *= round_trip_root;
        let sensitivity =
            metallic_film_sensitivity(channel, f64::from(order) * optical_path_difference);
        reflectance +=
            coefficient * 2.0 * (accumulator.re * sensitivity.re + accumulator.im * sensitivity.im);
        accumulator = accumulator.multiply(phasor);
    }
    reflectance
}

fn metallic_film_sensitivity(channel: usize, optical_path_difference: f64) -> MetallicFilmComplex {
    let x = saturate(TAU * optical_path_difference / 60_000.0) * 511.0;
    let index = (x as usize).min(511);
    let next = (index + 1).min(511);
    let fraction = x - index as f64;
    let first = THIN_FILM_CMF[index];
    let second = THIN_FILM_CMF[next];
    MetallicFilmComplex {
        re: f64::from(f32::from_bits(first[channel])) * (1.0 - fraction)
            + f64::from(f32::from_bits(second[channel])) * fraction,
        im: f64::from(f32::from_bits(first[channel + 3])) * (1.0 - fraction)
            + f64::from(f32::from_bits(second[channel + 3])) * fraction,
    }
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn standalone_metallic_fss(params: &BsdfParams, metallic: MetallicParameters) -> DVec3 {
    match metallic.fresnel {
        MetallicFresnel::F82 => {
            let f0 = standalone_color(params).clamp(DVec3::ZERO, DVec3::ONE);
            let tint = DVec3::from_array(metallic.edge_tint).clamp(DVec3::ZERO, DVec3::ONE);
            f82_fss(f0, f82_tint_b(f0, tint))
        }
        MetallicFresnel::PhysicalConductor => {
            let ior = DVec3::from_array(metallic.conductor_ior).max(DVec3::ZERO);
            let extinction = DVec3::from_array(metallic.extinction).max(DVec3::ZERO);
            let f0 = physical_conductor_fresnel(1.0, ior, extinction);
            let f82 = physical_conductor_fresnel(1.0 / 7.0, ior, extinction);
            f82_fss(f0, f82_tint_b_from_f82(f0, f82)).clamp(DVec3::ZERO, DVec3::ONE)
        }
    }
}

fn f82_tint_b_from_f82(f0: DVec3, f82: DVec3) -> DVec3 {
    let f: f64 = 6.0 / 7.0;
    let f5 = f.powi(5);
    let schlick = f0.lerp(DVec3::ONE, f5);
    (schlick - f82) * (7.0 / (f5 * f))
}

fn standalone_metallic_gain(
    params: &BsdfParams,
    metallic: MetallicParameters,
    layers: &Layers,
) -> DVec3 {
    if metallic.distribution != MicrofacetDistribution::MultiGgx {
        return DVec3::ONE;
    }
    let microfacet = standalone_metallic_setup_microfacet(params, layers, metallic.distribution);
    let roughness = (microfacet.alpha_x * microfacet.alpha_y).sqrt().sqrt();
    let energy = ggx_energy(roughness, microfacet.normal.dot(layers.view));
    let missing = (1.0 - energy) / energy;
    multiscatter_gain(
        standalone_metallic_fss(params, metallic),
        ggx_energy_average(roughness),
        missing,
    )
}

fn standalone_metallic_sample(
    params: &BsdfParams,
    layers: &Layers,
    geometric: DVec3,
    metallic: MetallicParameters,
    random: [f64; 3],
) -> StandaloneSample {
    let microfacet = standalone_metallic_microfacet(params, layers, metallic.distribution);
    let cosine_i = microfacet.normal.dot(layers.view);
    if cosine_i <= 0.0 {
        return StandaloneSample::Absorbed;
    }
    let singular = microfacet.alpha_x * microfacet.alpha_y <= GLASS_SINGULAR_ALPHA_SQUARED;
    let half = if singular {
        microfacet.normal
    } else {
        sample_standalone_half(microfacet, layers.view, random[0], random[1])
    };
    let cosine_hi = half.dot(layers.view);
    let fresnel = standalone_metallic_fresnel(params, metallic, cosine_hi);
    if fresnel.max_element() <= 0.0 {
        return StandaloneSample::Absorbed;
    }
    let direction = 2.0 * cosine_hi * half - layers.view;
    if geometric.dot(direction) < 0.0 || microfacet.normal.dot(direction) < 0.0 {
        return StandaloneSample::Absorbed;
    }
    if singular {
        StandaloneSample::Delta {
            direction,
            weight: fresnel * standalone_metallic_gain(params, metallic, layers),
            ray: RayKind::Glossy,
            branch_pdf: 1.0,
        }
    } else {
        StandaloneSample::Continuous {
            direction,
            ray: RayKind::Glossy,
        }
    }
}

fn standalone_refraction_sample(
    params: &BsdfParams,
    layers: &Layers,
    geometric: DVec3,
    distribution: MicrofacetDistribution,
    random: [f64; 3],
) -> StandaloneSample {
    let microfacet = standalone_refraction_microfacet(params, layers, distribution);
    let cosine_i = microfacet.normal.dot(layers.view);
    if cosine_i <= 0.0 {
        return StandaloneSample::Absorbed;
    }
    let eta = standalone_eta(params);
    let singular = microfacet.alpha_x * microfacet.alpha_y <= GLASS_SINGULAR_ALPHA_SQUARED;
    let half = if singular {
        microfacet.normal
    } else {
        sample_standalone_half(microfacet, layers.view, random[0], random[1])
    };
    let cosine_hi = half.dot(layers.view);
    let (fresnel, cosine_ho) = dielectric_fresnel(cosine_hi, eta);
    if fresnel >= 1.0 {
        return StandaloneSample::Absorbed;
    }
    let direction = (layers.view.dot(half) / eta + cosine_ho) * half - layers.view / eta;
    let cosine_o = microfacet.normal.dot(direction);
    let geometric_cosine_o = geometric.dot(direction);
    let geometric_cosine_is_negative = matches!(
        geometric_cosine_o.partial_cmp(&0.0),
        Some(std::cmp::Ordering::Less)
    );
    let cosine_is_negative = matches!(cosine_o.partial_cmp(&0.0), Some(std::cmp::Ordering::Less));
    if !geometric_cosine_is_negative || !cosine_is_negative {
        return StandaloneSample::Absorbed;
    }
    if singular || (eta - 1.0).abs() < 1.0e-4 {
        StandaloneSample::Delta {
            direction,
            weight: standalone_color(params),
            ray: RayKind::Transmission,
            branch_pdf: 1.0,
        }
    } else {
        StandaloneSample::Continuous {
            direction,
            ray: RayKind::Transmission,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Material;

    #[test]
    fn anisotropic_ggx_axes_and_rotation_follow_the_closure_model() {
        let mut params = BsdfParams::from_material(&Material::default(), DVec3::Z);
        params.anisotropy = 0.9;
        let unlinked = ggx_microfacet(&params, 0.35, DVec3::Z, DVec3::Z);
        assert!(!unlinked.anisotropic);
        assert_eq!(unlinked.alpha_x, unlinked.alpha_y);

        params.anisotropic_tangent = Some(DVec3::X);
        let principled = ggx_microfacet(&params, 0.35, DVec3::Z, DVec3::Z);
        let alpha = 0.35_f64.powi(2);
        let aspect = (1.0_f64 - 0.9 * 0.9).sqrt();
        assert!(principled.anisotropic);
        assert!((principled.alpha_x - alpha / aspect).abs() < 1.0e-12);
        assert!((principled.alpha_y - alpha * aspect).abs() < 1.0e-12);

        params.anisotropic_rotation = 0.25;
        let rotated = ggx_microfacet(&params, 0.35, DVec3::Z, DVec3::Z);
        assert!((rotated.tangent - DVec3::Y).length() < 1.0e-12);

        params.anisotropy_model = AnisotropyModel::Glossy;
        params.anisotropy = 0.5;
        params.anisotropic_rotation = 0.0;
        let glossy = ggx_microfacet(&params, 0.35, DVec3::Z, DVec3::Z);
        assert!((glossy.alpha_x - alpha * 0.5).abs() < 1.0e-12);
        assert!((glossy.alpha_y - alpha / 0.5).abs() < 1.0e-12);

        let view = DVec3::new(0.5, 0.0, 3.0_f64.sqrt() / 2.0);
        let half = sample_specular_ggx_half(glossy, DVec3::Z, view, 0.2, 0.7);
        assert!(view.dot(half) > 0.0);
        assert!(DVec3::Z.dot(half) > 0.0);
    }

    #[test]
    fn disabled_standalone_subsurface_keeps_the_diffuse_color() {
        let params = BsdfParams {
            model: BsdfModel::Subsurface,
            base_color: [0.72, 0.31, 0.12, 1.0],
            subsurface_radius: [0.85, 0.31, 0.12],
            subsurface_scale: 0.24,
            subsurface_method: SubsurfaceMethod::Burley,
            subsurface_weight: 0.0,
            ..BsdfParams::from_material(&Material::default(), DVec3::Z)
        };
        let (subsurface, diffuse, subsurface_probability, diffuse_probability, total) =
            standalone_subsurface_weights(&params);
        assert_eq!(subsurface, DVec3::ZERO);
        assert_eq!(diffuse, DVec3::new(0.72, 0.31, 0.12));
        assert_eq!(subsurface_probability, 0.0);
        assert_eq!(diffuse_probability, 1.0);
        assert!((total - 1.15 / 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn direct_eval_capability_tracks_bssrdf_only_and_diffuse_fallback() {
        let mut params = BsdfParams::from_material(&Material::default(), DVec3::Z);
        params.base_color = [0.72, 0.31, 0.12, 1.0];
        params.subsurface_radius = [0.85, 0.31, 0.12];
        params.subsurface_scale = 0.24;
        params.subsurface_weight = 1.0;
        params.model = BsdfModel::Subsurface;
        let layer_set = layers(&params, DVec3::Z, DVec3::Z);
        assert!(!bsdf_has_direct_eval(&params, &layer_set));

        params.subsurface_radius[2] = 1.0e-9;
        let layer_set = layers(&params, DVec3::Z, DVec3::Z);
        assert!(bsdf_has_direct_eval(&params, &layer_set));

        params.subsurface_radius = [0.85, 0.31, 0.12];
        params.thin_wall = true;
        let layer_set = layers(&params, DVec3::Z, DVec3::Z);
        assert!(bsdf_has_direct_eval(&params, &layer_set));

        params.model = BsdfModel::Principled;
        params.thin_wall = false;
        params.reflectance = 0.0;
        params.specular_ior_level = 0.0;
        params.coat_weight = 0.0;
        params.sheen_weight = 0.0;
        params.transmission = 0.0;
        params.roughness = 0.0;
        let layer_set = layers(&params, DVec3::Z, DVec3::Z);
        assert!(bsdf_has_direct_eval(&params, &layer_set));

        params.subsurface_radius[0] = 1.0e-9;
        let layer_set = layers(&params, DVec3::Z, DVec3::Z);
        assert!(bsdf_has_direct_eval(&params, &layer_set));
    }

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

    #[test]
    fn concentric_disk_stays_in_the_unit_disk_and_maps_the_centre_to_the_origin() {
        for (u, v) in [
            (0.0, 0.0),
            (0.999, 0.5),
            (0.5, 0.001),
            (0.25, 0.75),
            (0.9, 0.1),
        ] {
            let (x, y) = concentric_disk(u, v);
            assert!(
                x * x + y * y <= 1.0 + 1.0e-12,
                "({u}, {v}) maps to ({x}, {y})"
            );
        }
        assert_eq!(concentric_disk(0.5, 0.5), (0.0, 0.0));
    }

    #[test]
    fn sheen_frame_is_orthonormal_with_bitangent_along_normal_cross_view() {
        let normal = DVec3::new(0.2, -0.3, 0.9).normalize();
        let view = DVec3::new(0.5, 0.1, 0.7).normalize();
        let (tangent, bitangent) = sheen_frame(normal, view);
        assert!((tangent.length() - 1.0).abs() < 1.0e-12);
        assert!((bitangent.length() - 1.0).abs() < 1.0e-12);
        assert!(tangent.dot(normal).abs() < 1.0e-12);
        assert!(bitangent.dot(normal).abs() < 1.0e-12);
        assert!(tangent.dot(bitangent).abs() < 1.0e-12);
        assert!((bitangent - normal.cross(view).normalize()).length() < 1.0e-12);
    }

    #[test]
    fn sheen_frame_falls_back_to_an_orthonormal_basis_at_normal_incidence() {
        let (tangent, bitangent) = sheen_frame(DVec3::Z, DVec3::Z);
        assert!((tangent.length() - 1.0).abs() < 1.0e-12);
        assert!((bitangent.length() - 1.0).abs() < 1.0e-12);
        assert!(tangent.dot(DVec3::Z).abs() < 1.0e-12);
        assert!(bitangent.dot(DVec3::Z).abs() < 1.0e-12);
        assert!(tangent.dot(bitangent).abs() < 1.0e-12);
    }

    /// Midpoint quadrature of `f` over the upper hemisphere with the solid-angle measure.
    fn hemisphere_integral(f: impl Fn(DVec3) -> f64) -> f64 {
        const THETA_STEPS: usize = 400;
        const PHI_STEPS: usize = 800;
        let mut total = 0.0;
        for i in 0..THETA_STEPS {
            let theta = (i as f64 + 0.5) / THETA_STEPS as f64 * FRAC_PI_2;
            for j in 0..PHI_STEPS {
                let phi = (j as f64 + 0.5) / PHI_STEPS as f64 * std::f64::consts::TAU;
                let direction = DVec3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                );
                let measure = theta.sin()
                    * (FRAC_PI_2 / THETA_STEPS as f64)
                    * (std::f64::consts::TAU / PHI_STEPS as f64);
                total += f(direction) * measure;
            }
        }
        total
    }

    fn sheen_params(weight: f64, roughness: f64, tint: [f64; 3]) -> BsdfParams {
        let material = crate::model::Material {
            base_color: [0.0, 0.0, 0.0, 1.0],
            metallic: 0.0,
            roughness: 0.24,
            ..crate::model::Material::default()
        };
        let mut params = BsdfParams::from_material(&material, DVec3::Z);
        params.specular_ior_level = 0.0;
        params.sheen_weight = weight;
        params.sheen_roughness = roughness;
        params.sheen_tint = tint;
        params
    }

    #[test]
    fn sheen_density_integrates_to_one_over_the_hemisphere() {
        let view = DVec3::new(0.5, 0.0, 0.5_f64.sqrt());
        for roughness in [0.3, 0.5, 1.0] {
            let params = sheen_params(1.0, roughness, [1.0; 3]);
            let lobe = sheen_lobe(&params, DVec3::Z, view);
            let total = hemisphere_integral(|light| lobe.density(light));
            assert!(
                (total - 1.0).abs() < 1.0e-2,
                "roughness {roughness}: density integrates to {total}"
            );
        }
    }

    #[test]
    fn sheen_samples_follow_the_density() {
        // The fraction of LTC samples inside a cap equals the density's integral over the cap.
        const SIDE: usize = 400;
        let view = DVec3::new(0.5, 0.0, 0.5_f64.sqrt());
        let params = sheen_params(1.0, 0.5, [1.0; 3]);
        let shading = layers(&params, DVec3::Z, view);
        for cap in [0.6_f64, 1.0] {
            let inside = (0..SIDE)
                .flat_map(|i| (0..SIDE).map(move |j| (i, j)))
                .filter(|&(i, j)| {
                    let u = (i as f64 + 0.5) / SIDE as f64;
                    let v = (j as f64 + 0.5) / SIDE as f64;
                    shading.sample_sheen(u, v).z >= cap.cos()
                })
                .count();
            let sampled = inside as f64 / (SIDE * SIDE) as f64;
            let lobe = sheen_lobe(&params, DVec3::Z, view);
            let expected = hemisphere_integral(|light| {
                if light.z >= cap.cos() {
                    lobe.density(light)
                } else {
                    0.0
                }
            });
            assert!(
                (sampled - expected).abs() < 5.0e-3,
                "cap {cap}: sampled {sampled} vs density {expected}"
            );
        }
    }

    #[test]
    fn sheen_layer_keeps_one_minus_the_largest_channel_albedo() {
        let view = DVec3::new(0.5, 0.0, 0.5_f64.sqrt());
        let params = sheen_params(0.6, 0.3, [1.0, 0.5, 0.25]);
        let albedo = sheen_ltc(0.3, view.z).albedo;
        let layer = sheen_lobe(&params, DVec3::Z, view).layer_weight();
        assert!((layer - (1.0 - 0.6 * albedo)).abs() < 1.0e-12);
        let off = sheen_params(0.0, 0.3, [1.0; 3]);
        assert_eq!(sheen_lobe(&off, DVec3::Z, view).layer_weight(), 1.0);
        assert_eq!(
            sheen_coefficient(&off, &layers(&off, DVec3::Z, view), view),
            DVec3::ZERO
        );
    }

    #[test]
    fn sheen_normal_mixes_toward_the_coat_normal_by_coat_weight() {
        let view = DVec3::new(0.5, 0.0, 0.5_f64.sqrt());
        let mut params = sheen_params(1.0, 0.5, [1.0; 3]);
        params.coat_normal = Some(DVec3::X);
        params.coat_weight = 0.5;
        // normalize(mix(N, coat normal, 0.5)) with N = +Z, as Cycles' sheen N.
        let expected = DVec3::Z.lerp(DVec3::X, 0.5).normalize();
        let shading = layers(&params, DVec3::Z, view);
        assert!((shading.sheen.normal - expected).length() < 1.0e-12);
        // Without a coat the sheen keeps the shading normal.
        params.coat_weight = 0.0;
        let shading = layers(&params, DVec3::Z, view);
        assert!((shading.sheen.normal - DVec3::Z).length() < 1.0e-12);
    }

    #[test]
    fn glossy_blur_raises_standalone_refraction_alpha_and_keeps_a_larger_alpha() {
        let mut params = BsdfParams::from_material(&Material::default(), DVec3::Z);
        params.model = BsdfModel::Refraction(MicrofacetDistribution::Ggx);
        params.roughness = 0.13;
        let sharp = layers(&params, DVec3::Z, DVec3::Z);
        let alpha = |layers: &Layers| {
            standalone_refraction_microfacet(&params, layers, MicrofacetDistribution::Ggx).alpha_x
        };
        assert!((alpha(&sharp) - 0.13_f64.powi(2)).abs() < 1.0e-12);
        assert_eq!(alpha(&sharp.blurred(0.5)), 0.5);
        assert!((alpha(&sharp.blurred(0.001)) - 0.13_f64.powi(2)).abs() < 1.0e-12);
    }

    #[test]
    fn glossy_blur_raises_the_metallic_sampling_alpha_but_not_its_multiscatter_gain() {
        let mut params = BsdfParams::from_material(&Material::default(), DVec3::Z);
        params.roughness = 0.3;
        let metallic = MetallicParameters {
            distribution: MicrofacetDistribution::MultiGgx,
            fresnel: MetallicFresnel::PhysicalConductor,
            edge_tint: [1.0; 3],
            conductor_ior: [1.5; 3],
            extinction: [1.0; 3],
        };
        params.model = BsdfModel::Metallic(metallic);
        let view = DVec3::new(0.6, 0.0, 0.8);
        let sharp = layers(&params, DVec3::Z, view);
        let blurred = sharp.blurred(0.5);
        let sampling = standalone_metallic_microfacet(&params, &blurred, metallic.distribution);
        assert_eq!(sampling.alpha_x, 0.5);
        assert_eq!(sampling.alpha_y, 0.5);
        assert_eq!(
            standalone_metallic_gain(&params, metallic, &sharp),
            standalone_metallic_gain(&params, metallic, &blurred)
        );
    }
    #[test]
    fn shadow_terminator_shading_offset_uses_cycles_frequency_multiplier() {
        let params = BsdfParams::from_material(&Material::default(), DVec3::Z);
        let cosine = 0.5;
        let ordinary = layers(&params, DVec3::Z, DVec3::Z);
        assert_eq!(ordinary.shadow_terminator_multiplier(cosine), 1.0);

        let raw_offset = 0.5;
        let offset = ordinary.with_shadow_terminator_shading_offset(raw_offset);
        let frequency_multiplier = f64::from(1.0_f32 / (1.0_f32 - 0.5_f32 * raw_offset as f32));
        assert_eq!(
            offset.shadow_terminator_frequency_multiplier,
            frequency_multiplier
        );
        let expected = (cosine.acos() * frequency_multiplier).cos().max(0.0) / cosine;
        assert!((offset.shadow_terminator_multiplier(cosine) - expected).abs() < 1.0e-12);
        assert_eq!(
            ordinary
                .with_shadow_terminator_shading_offset(0.0)
                .shadow_terminator_multiplier(cosine),
            1.0
        );
    }
}

#[cfg(test)]
mod glass_tests {
    use super::*;
    use crate::model::Material;

    // Geometry of tests/fixtures/native_oracle/glass/glass_plane_probe.py: a glass plane at z = 0 seen
    // from 45° through a 16 × 16 orthographic image of 0.2 units, with the central 8 × 8 pixels
    // averaged. A disk emitter of radius 1.2 and power 1000 W lies 4 units away: the front light on the
    // mirror of the view, the back light on the refraction of the view below the plane.
    const VIEW_THETA: f64 = FRAC_PI_4;
    const LIGHT_DISTANCE: f64 = 4.0;
    const RADIUS: f64 = 1.2;
    const POWER: f64 = 1000.0;
    const ORTHO_SCALE: f64 = 0.2;
    /// Quadrature of the disk used for the oracle comparisons: radial and angular midpoints. The midpoint rule is
    /// second order and the lobe is narrow, so the grid was chosen by a convergence run (a numpy port of this
    /// integral, the same receivers, grids 64 × 128, 128 × 256, 256 × 512, 512 × 1024): at roughness 0.2 the back
    /// patch mean is 66.847, 66.681, 66.640, 66.630, the error dropping by about 4 per doubling. At 256 × 512 it is
    /// 0.0103 from 512 × 1024, 0.68 of the Cycles stderr (0.0152), and each rough oracle case is within 1 stderr.
    const RADIAL: usize = 256;
    const ANGULAR: usize = 512;

    fn view() -> DVec3 {
        DVec3::new(VIEW_THETA.sin(), 0.0, VIEW_THETA.cos())
    }

    /// A Principled glass (Transmission Weight 1, white base) as `BsdfParams::from_material` builds it.
    fn principled_glass(ior: f64, roughness: f64) -> BsdfParams {
        let material = Material {
            base_color: [1.0, 1.0, 1.0, 1.0],
            roughness,
            ior,
            transmission: 1.0,
            ..Material::default()
        };
        BsdfParams::from_material(&material, DVec3::Z)
    }

    /// The Glass BSDF node: its Color is both the reflection and the transmission tint.
    fn glass_node(color: [f64; 3], ior: f64, roughness: f64) -> BsdfParams {
        let mut params = principled_glass(ior, roughness);
        params.base_color = [color[0], color[1], color[2], 1.0];
        params.glass_reflection_tint = color;
        params.glass_transmission_tint = color;
        params
    }
    #[test]
    fn cycles_parity_thin_glass_caustics_gate_both_fresnel_branches() {
        let mut params = principled_glass(1.5, 0.5);
        params.thin_wall = true;
        let normal = DVec3::Z;
        let view = DVec3::Z;

        let disabled = layers(&params, normal, view).with_caustics(&params, false, false);
        let Some(disabled_thin) = disabled.glass.and_then(|glass| glass.thin) else {
            panic!("Thin Wall must retain its Principled glass setup");
        };
        assert_eq!(disabled_thin.reflectance, DVec3::ZERO);
        assert_eq!(disabled_thin.transmittance, DVec3::ZERO);
        assert_eq!(
            disabled_thin.selection_weight().to_bits(),
            0.0_f64.to_bits()
        );

        let transmission_only = layers(&params, normal, view).with_caustics(&params, false, true);
        let Some(transmission_only) = transmission_only.glass.and_then(|glass| glass.thin) else {
            panic!("refractive caustics must retain thin-glass transmission");
        };
        assert_eq!(transmission_only.reflectance, DVec3::ZERO);
        assert!(transmission_only.transmittance.max_element() > 0.0);

        let reflection_only = layers(&params, normal, view).with_caustics(&params, true, false);
        let Some(reflection_only) = reflection_only.glass.and_then(|glass| glass.thin) else {
            panic!("reflective caustics must retain thin-glass reflection");
        };
        assert!(reflection_only.reflectance.max_element() > 0.0);
        assert_eq!(reflection_only.transmittance, DVec3::ZERO);
    }

    #[test]
    fn cycles_parity_thin_glass_caustics_filter_preserves_glossy_blur() {
        let mut params = principled_glass(1.5, 0.2);
        params.thin_wall = true;
        let filtered = layers(&params, DVec3::Z, DVec3::Z)
            .blurred(0.4)
            .with_caustics(&params, false, true);
        let Some(glass) = filtered.glass else {
            panic!("refractive caustics must retain thin-glass transmission");
        };
        assert_eq!(glass.alpha.to_bits(), 0.4_f64.to_bits());
        let Some(thin) = glass.thin else {
            panic!("Thin Wall must retain its two glass lobes");
        };
        assert_eq!(thin.transmission_alpha.to_bits(), 0.4_f64.to_bits());
    }

    /// The light centre of the probe: 4 units along the mirror of the view (front), or along the
    /// refraction of the view below the plane for a relative IOR `ior` (back).
    fn light_centre(ior: f64, back: bool) -> DVec3 {
        if !back {
            return LIGHT_DISTANCE * DVec3::new(-VIEW_THETA.sin(), 0.0, VIEW_THETA.cos());
        }
        let sin_t = VIEW_THETA.sin() / ior;
        LIGHT_DISTANCE * DVec3::new(-sin_t, 0.0, -(1.0 - sin_t * sin_t).sqrt())
    }

    /// Midpoint quadrature of `f` over the unit sphere, `Σ f(ω) dω`.
    fn sphere_integral(n_theta: usize, n_phi: usize, mut f: impl FnMut(DVec3) -> f64) -> f64 {
        let d_theta = PI / n_theta as f64;
        let d_phi = std::f64::consts::TAU / n_phi as f64;
        let mut total = 0.0;
        for i in 0..n_theta {
            let theta = (i as f64 + 0.5) * d_theta;
            let (sin_t, cos_t) = theta.sin_cos();
            for j in 0..n_phi {
                let phi = (j as f64 + 0.5) * d_phi;
                let direction = DVec3::new(sin_t * phi.cos(), sin_t * phi.sin(), cos_t);
                total += f(direction) * sin_t * d_theta * d_phi;
            }
        }
        total
    }

    /// A deterministic uniform stream on `[0, 1)` for the Monte Carlo checks.
    fn uniform_stream(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        }
    }

    /// Cycles' single-precision disk emitter radiance: `P / (π · area)`.
    fn disk_radiance() -> f64 {
        POWER / (PI * PI * RADIUS * RADIUS)
    }

    /// Whether `light` lies in the domain the VNDF sampler draws: its half vector on the normal's side
    /// (`n · H >= 0`) and visible from the view (`H · wi >= 0`). Cycles' eval also carries half vectors
    /// outside it (`bsdf_D` squares `cos_NH` and `microfacet_eval` takes `fabs(cos_HI)`), which the
    /// sampler's `max(0, h.z)` and visibility never produce, so an integral over the sphere is not the
    /// sampler's.
    fn sampler_domain(lobe: &GlassLobe, normal: DVec3, view: DVec3, light: DVec3) -> bool {
        let unnormalised = if normal.dot(light) < 0.0 {
            -(view + light * lobe.ior)
        } else {
            view + light
        };
        let half = unnormalised.normalize_or_zero();
        normal.dot(half) >= 0.0 && view.dot(half) >= 0.0
    }

    #[test]
    fn glass_density_integrates_to_the_accepted_draws() {
        let params = principled_glass(1.5, 0.5);
        let shading = layers(&params, DVec3::Z, view());
        let Some(lobe) = glass_lobe(&params, DVec3::Z, view(), true, true) else {
            panic!("a glass plane has a glass lobe");
        };
        // Over the sampler's domain the density is the density of the accepted rough draws.
        let density = sphere_integral(400, 800, |light| {
            if sampler_domain(&lobe, DVec3::Z, view(), light) {
                glass_eval(&lobe, DVec3::Z, view(), light).1
            } else {
                0.0
            }
        });
        let mut next = uniform_stream(7);
        let draws = 200_000_usize;
        let mut accepted = 0_usize;
        for _ in 0..draws {
            let rands = [next(), next(), next()];
            if matches!(
                sample_glass(&params, &shading, DVec3::Z, 1.0, rands),
                GlassSample::Rough { .. }
            ) {
                accepted += 1;
            }
        }
        let share = accepted as f64 / draws as f64;
        let stderr = (share * (1.0 - share) / draws as f64).sqrt();
        assert!(
            (density - share).abs() <= 4.0 * stderr + 1.0e-3,
            "density integrates to {density}, accepted share {share}"
        );
    }

    #[test]
    fn white_glass_visible_albedo_matches_the_ggx_glass_energy_table() {
        // The GGX_GLASS tables are Cycles' `average(eval) / pdf` over its sampler, so E is the eval's
        // integral over the sampler's domain. At a table node (roughness 6/15, cos 11/15, z 8/15) the
        // two agree to 1e-4, and the gain 1 / E then makes white glass integrate to 1 over that domain.
        let (roughness, mu, z): (f64, f64, f64) = (6.0 / 15.0, 11.0 / 15.0, 8.0 / 15.0);
        let ior = (1.0 + z * z) / (1.0 - z * z);
        let view = DVec3::new((1.0 - mu * mu).sqrt(), 0.0, mu);
        let params = principled_glass(ior, roughness);
        let Some(lobe) = glass_lobe(&params, DVec3::Z, view, true, true) else {
            panic!("a glass plane has a glass lobe");
        };
        let albedo = sphere_integral(400, 800, |light| {
            if sampler_domain(&lobe, DVec3::Z, view, light) {
                glass_eval(&lobe, DVec3::Z, view, light).0.element_sum() / 3.0
            } else {
                0.0
            }
        });
        let energy = ggx_glass_energy(roughness, mu, z);
        assert!(
            (albedo / lobe.gain.x - energy).abs() < 2.0e-3,
            "eval integral {} vs GGX_GLASS_E {energy}",
            albedo / lobe.gain.x
        );
        assert!((albedo - 1.0).abs() < 2.0e-3, "white glass albedo {albedo}");
    }

    #[test]
    fn glass_transmission_is_reciprocal_up_to_the_squared_ior() {
        // Walter's reciprocity f(wi, wo) / η_o² = f(wo, wi) / η_i², with the gain set to one (the gain
        // follows the view, so it is not reciprocal). The back lobe is the front one seen from below.
        let ior = 1.5;
        let front = GlassLobe {
            thin_film_thickness: 0.0,
            thin_film_ior: 1.33,
            ior,
            alpha: 0.25,
            f0: DVec3::splat(dielectric_f0(ior, 0.5)),
            reflection_tint: DVec3::ONE,
            transmission_tint: DVec3::ONE,
            gain: DVec3::ONE,
            selection_weight: 1.0,
            singular: false,
            straight: false,
            thin: None,
        };
        let back = GlassLobe {
            ior: 1.0 / ior,
            ..front
        };
        let directions = [
            (DVec3::new(0.3, 0.1, 0.95), DVec3::new(-0.3, 0.1, -0.9)),
            (DVec3::new(0.7, 0.0, 0.7), DVec3::new(-0.5, 0.2, -0.8)),
            (DVec3::new(0.9, -0.2, 0.4), DVec3::new(-0.2, -0.1, -1.0)),
        ];
        for (view, light) in directions {
            let view = view.normalize();
            let light = light.normalize();
            let (value_front, _) = glass_eval(&front, DVec3::Z, view, light);
            let f_front = value_front / DVec3::Z.dot(light).abs();
            let (value_back, _) = glass_eval(&back, -DVec3::Z, light, view);
            let f_back = value_back / (-DVec3::Z).dot(view).abs();
            assert!(f_back.max_element() > 0.0);
            let difference = (f_front / (ior * ior) - f_back).abs().max_element();
            assert!(
                difference <= 1.0e-9 * f_back.max_element(),
                "view {view:?} light {light:?}: {f_front:?} vs {f_back:?}"
            );
        }
    }

    #[test]
    fn sharp_glass_draws_the_fresnel_split_with_unit_weight() {
        let params = principled_glass(1.5, 0.0);
        let shading = layers(&params, DVec3::Z, view());
        let fresnel = dielectric_fresnel(view().z, 1.5).0;
        let draws = 10_000_usize;
        let mut reflected = 0_usize;
        for k in 0..draws {
            let u = (k as f64 + 0.5) / draws as f64;
            let GlassSample::Singular {
                direction, weight, ..
            } = sample_glass(&params, &shading, DVec3::Z, 1.0, [0.5, 0.5, u])
            else {
                panic!("a sharp glass draw is singular");
            };
            // Reflection divides R by its share, refraction T by its share: both weigh one.
            assert!(
                (weight - DVec3::ONE).abs().max_element() < 1.0e-3,
                "{weight:?}"
            );
            if direction.z > 0.0 {
                reflected += 1;
            } else {
                // Snell: the tangential part of the refracted direction is sin θ / η.
                let tangential = (direction - DVec3::Z * direction.z).length();
                assert!((tangential - view().x / 1.5).abs() < 1.0e-9);
            }
        }
        let share = reflected as f64 / draws as f64;
        assert!(
            (share - fresnel).abs() < 1.0e-3,
            "reflected {share} vs Fresnel {fresnel}"
        );
    }

    /// Patch receivers of the probe: the central 8 × 8 pixels, mirror symmetric in y, so the
    /// lower four rows stand for their mirror images (`2 ×`).
    fn patch_receivers() -> Vec<DVec3> {
        let pixel = ORTHO_SCALE / 16.0;
        let x_step = pixel / VIEW_THETA.cos();
        let mut receivers = Vec::with_capacity(32);
        for i in 0..8 {
            for j in 0..4 {
                let x = (f64::from(i) - 3.5) * x_step;
                let y = (f64::from(j) - 3.5) * pixel;
                receivers.push(DVec3::new(x, y, 0.0));
            }
        }
        receivers
    }

    /// Radiance from the disk at `centre` that a delta lobe along `direction` from `receiver` sees.
    fn disk_hit(receiver: DVec3, direction: DVec3, centre: DVec3) -> f64 {
        let normal = -centre.normalize();
        let facing = direction.dot(normal);
        if facing >= 0.0 {
            return 0.0;
        }
        let t = (centre - receiver).dot(normal) / facing;
        if t <= 0.0 || (receiver + direction * t - centre).length() > RADIUS {
            return 0.0;
        }
        disk_radiance()
    }

    /// The probe's pixel mean of a sharp glass plane: each receiver's delta lobe toward the light
    /// centre, weighted by `weight` (the Fresnel reflectance or transmittance) and the lobe's tint.
    fn sharp_patch_mean(tint: DVec3, weight: f64, centre: DVec3) -> DVec3 {
        let direction = centre.normalize();
        let total = patch_receivers().iter().fold(DVec3::ZERO, |sum, receiver| {
            sum + 2.0 * tint * weight * disk_hit(*receiver, direction, centre)
        });
        total / 64.0
    }

    /// Single-bounce integral over the disk at `centre` for a receiver, with `|cos|` on both sides.
    fn disk_integral(
        params: &BsdfParams,
        shading: &Layers,
        receiver: DVec3,
        centre: DVec3,
        radial: usize,
        angular: usize,
    ) -> DVec3 {
        let toward_receiver = -centre.normalize();
        let u = toward_receiver.cross(DVec3::Y).normalize();
        let w = toward_receiver.cross(u);
        let radiance = disk_radiance();
        let mut total = DVec3::ZERO;
        for i in 0..radial {
            let r = (i as f64 + 0.5) / radial as f64;
            for j in 0..angular {
                let phi = (j as f64 + 0.5) / angular as f64 * std::f64::consts::TAU;
                let point = centre + RADIUS * (r * phi.cos() * u + r * phi.sin() * w);
                let offset = point - receiver;
                let distance = offset.length();
                let direction = offset / distance;
                let cos_receiver = direction.dot(DVec3::Z).abs();
                let cos_emitter = (-direction).dot(toward_receiver).max(0.0);
                let area =
                    r / radial as f64 * (std::f64::consts::TAU / angular as f64) * RADIUS * RADIUS;
                let weight = area * cos_emitter / (distance * distance) * radiance * cos_receiver;
                total += glass_coefficient(params, shading, direction) * weight;
            }
        }
        total
    }

    /// The probe's pixel mean of a rough glass plane (the lobe is integrated over the disk).
    fn rough_patch_mean(
        params: &BsdfParams,
        centre: DVec3,
        radial: usize,
        angular: usize,
    ) -> DVec3 {
        let shading = layers(params, DVec3::Z, view());
        let total = patch_receivers().iter().fold(DVec3::ZERO, |sum, receiver| {
            sum + 2.0 * disk_integral(params, &shading, *receiver, centre, radial, angular)
        });
        total / 64.0
    }

    /// The Fresnel reflectance and transmittance of the sharp glass at the probe's view.
    fn sharp_split(ior: f64) -> (f64, f64) {
        let reflectance = dielectric_fresnel(view().z, ior).0;
        (reflectance, 1.0 - reflectance)
    }

    fn assert_matches_cycles(name: &str, potter: DVec3, mean: [f64; 3], stderr: [f64; 3]) {
        for channel in 0..3 {
            let tolerance = 4.0 * stderr[channel];
            assert!(
                (potter[channel] - mean[channel]).abs() <= tolerance,
                "{name} channel {channel}: Potter {} vs Cycles {} (tolerance {tolerance})",
                potter[channel],
                mean[channel]
            );
        }
    }

    #[test]
    fn sharp_glass_plane_matches_cycles() {
        // tests/fixtures/native_oracle/glass/glass_plane_means.json, roughness 0.
        let white = DVec3::ONE;
        let (front, back) = sharp_split(1.5);
        assert_matches_cycles(
            "P_r0_ior150 front",
            sharp_patch_mean(white, front, light_centre(1.5, false)),
            [3.535_039_4; 3],
            [4.709_511_6e-05; 3],
        );
        assert_matches_cycles(
            "P_r0_ior150 back",
            sharp_patch_mean(white, back, light_centre(1.5, true)),
            [66.859_555; 3],
            [0.015_682_058; 3],
        );
        let (front, back) = sharp_split(1.45);
        assert_matches_cycles(
            "P_r0_ior145 front",
            sharp_patch_mean(white, front, light_centre(1.45, false)),
            [3.048_357_2; 3],
            [8.431_144_8e-05; 3],
        );
        assert_matches_cycles(
            "P_r0_ior145 back",
            sharp_patch_mean(white, back, light_centre(1.45, true)),
            [67.344_936; 3],
            [0.017_525_662; 3],
        );
    }

    #[test]
    fn rough_glass_plane_matches_cycles_at_roughness_0_2() {
        // tests/fixtures/native_oracle/glass/glass_plane_means.json, roughness 0.2.
        let white = principled_glass(1.5, 0.2);
        assert_matches_cycles(
            "P_r02_ior150 front",
            rough_patch_mean(&white, light_centre(1.5, false), RADIAL, ANGULAR),
            [3.365_433_6; 3],
            [2.087_910_2e-04; 3],
        );
        assert_matches_cycles(
            "P_r02_ior150 back",
            rough_patch_mean(&white, light_centre(1.5, true), RADIAL, ANGULAR),
            [66.64674; 3],
            [0.015_211_354; 3],
        );
        let mut tinted = principled_glass(1.5, 0.2);
        tinted.base_color = [0.8, 0.5, 0.3, 1.0];
        tinted.glass_transmission_tint = [0.8_f64, 0.5, 0.3].map(f64::sqrt);
        assert_matches_cycles(
            "P_r02_ior150_tint front",
            rough_patch_mean(&tinted, light_centre(1.5, false), RADIAL, ANGULAR),
            [3.365_523_7, 3.365_215_8, 3.364_953_5],
            [3.020_351_7e-04, 3.016_969_7e-04, 3.014_383_9e-04],
        );
        assert_matches_cycles(
            "P_r02_ior150_tint back",
            rough_patch_mean(&tinted, light_centre(1.5, true), RADIAL, ANGULAR),
            [59.604_277, 47.117_035, 36.493_874],
            [0.013_510_868, 0.010_680_354, 0.008_272_250_2],
        );
        let node = glass_node([0.8, 0.5, 0.3], 1.5, 0.2);
        assert_matches_cycles(
            "G_r02_ior150_tint front",
            rough_patch_mean(&node, light_centre(1.5, false), RADIAL, ANGULAR),
            [2.692_091_6, 1.682_310_6, 1.009_287_9],
            [1.648_273_9e-04, 1.030_548_6e-04, 6.184_239_8e-05],
        );
        assert_matches_cycles(
            "G_r02_ior150_tint back",
            rough_patch_mean(&node, light_centre(1.5, true), RADIAL, ANGULAR),
            [53.31221, 33.31528, 19.987_229],
            [0.012_167_949, 0.007_603_780_5, 0.004_561_878_2],
        );
    }

    #[test]
    fn rough_glass_plane_matches_cycles_at_roughness_0_2_ior_1_45() {
        // tests/fixtures/native_oracle/glass/glass_plane_means.json, roughness 0.2, IOR 1.45.
        let params = principled_glass(1.45, 0.2);
        assert_matches_cycles(
            "P_r02_ior145 front",
            rough_patch_mean(&params, light_centre(1.45, false), RADIAL, ANGULAR),
            [2.904_385_7; 3],
            [2.292_719_1e-04; 3],
        );
        assert_matches_cycles(
            "P_r02_ior145 back",
            rough_patch_mean(&params, light_centre(1.45, true), RADIAL, ANGULAR),
            [67.166_565; 3],
            [0.018_860_988; 3],
        );
    }

    #[test]
    fn rough_glass_plane_matches_cycles_at_roughness_0_5() {
        // tests/fixtures/native_oracle/glass/glass_plane_means.json, roughness 0.5.
        let params = principled_glass(1.5, 0.5);
        assert_matches_cycles(
            "P_r05_ior150 front",
            rough_patch_mean(&params, light_centre(1.5, false), RADIAL, ANGULAR),
            [1.155_500_4; 3],
            [8.010_481_1e-05; 3],
        );
        assert_matches_cycles(
            "P_r05_ior150 back",
            rough_patch_mean(&params, light_centre(1.5, true), RADIAL, ANGULAR),
            [60.844_538; 3],
            [0.007_913_841_4; 3],
        );
        let params = principled_glass(1.45, 0.5);
        assert_matches_cycles(
            "P_r05_ior145 front",
            rough_patch_mean(&params, light_centre(1.45, false), RADIAL, ANGULAR),
            [0.997_729_09; 3],
            [6.755_766_3e-05; 3],
        );
        assert_matches_cycles(
            "P_r05_ior145 back",
            rough_patch_mean(&params, light_centre(1.45, true), RADIAL, ANGULAR),
            [62.097_119; 3],
            [0.012_259_179; 3],
        );
    }

    #[test]
    fn glass_at_ior_one_refracts_straight_through_as_a_delta() {
        // Cycles' `bsdf_microfacet_sample` makes a refraction at |eta - 1| < 1e-4 singular, and its direction
        // is -wi whatever the half vector: the reflectance is zero, so every draw refracts.
        let params = principled_glass(1.0, 0.5);
        let shading = layers(&params, DVec3::Z, view());
        for u in [0.1, 0.5, 0.9] {
            let GlassSample::Singular {
                direction, weight, ..
            } = sample_glass(&params, &shading, DVec3::Z, 1.0, [0.3, 0.7, u])
            else {
                panic!("a refraction at an IOR of 1 is a delta");
            };
            assert!((direction + view()).length() < 1.0e-9, "{direction:?}");
            assert!(weight.x.is_finite() && weight.x > 0.0, "{weight:?}");
        }
    }
}

#[cfg(test)]
mod single_bounce_tests {
    use super::super::path::{Closures, HitClosure};
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
    /// oracles use `FINE` (384 × 768: 5e-7 and 1.0e-4 respectively). The specular lobe narrows
    /// as roughness falls, and `COARSE` overshoots it: against 768 × 1536 midpoints, the radius 1.2
    /// spec-only pixel is +0.01% at roughness 0.24, +0.10% at 0.12 and +0.17% at 0.10. Oracles below
    /// roughness 0.2 need `FINE` or finer. One 4096-sample Blender render at roughness 0.10 scatters
    /// by about 0.12% (16-seed stderr × 4).
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
        let closure = HitClosure::new(1.0, params, view, normal, true);
        let mut closures = Closures::new(vec![closure], normal, normal, true);
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
                let cos_emitter = (-direction).dot(toward_receiver).max(0.0);
                let area = r
                    * (1.0 / steps.radial as f64)
                    * (std::f64::consts::TAU / steps.angular as f64)
                    * RADIUS
                    * RADIUS;
                let weight = area * cos_emitter / (distance * distance) * radiance;
                total += closures.bsdf_eval(direction, None, RayKind::ALL_MASK) * weight;
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
    fn single_bounce_diffuse_roughness_matches_blender() {
        // tests/fixtures/native_oracle/principled/single_bounce_diffuse_roughness_means.json.
        let mut params = native([0.8, 0.5, 0.3, 1.0], 0.0);
        params.specular_ior_level = 0.0;
        for (name, roughness, mean, stderr) in [
            (
                "diffuse_roughness_r000",
                0.0,
                [
                    3.286_944_471_765_309_6,
                    2.054_340_287_926_607,
                    1.232_604_213_408_194_5,
                ],
                [
                    1.506_987_510_377_347e-4,
                    9.419_285_210_001_831e-5,
                    5.651_711_707_477_082e-5,
                ],
            ),
            (
                "diffuse_roughness_r050",
                0.5,
                [
                    2.417_323_027_970_269_3,
                    1.420_870_922_971_516_8,
                    0.831_206_352_915_614_8,
                ],
                [
                    1.327_489_060_053_661_6e-4,
                    7.681_696_530_561_993e-5,
                    4.463_368_364_025_854e-5,
                ],
            ),
            (
                "diffuse_roughness_r100",
                1.0,
                [
                    1.734_360_327_827_744_2,
                    0.924_668_854_568_153_6,
                    0.518_281_015_829_416,
                ],
                [
                    9.696_798_072_425_371e-5,
                    5.107_087_522_021_459e-5,
                    2.846_036_588_622_309e-5,
                ],
            ),
        ] {
            params.diffuse_roughness = roughness;
            assert_matches_oracle(name, &params, mean, stderr);
        }
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

    /// Sheen alone: the base diffuse is black and Specular IOR Level 0 removes the dielectric layer,
    /// so the sheen lobe is the only term (`sheen_only`).
    fn sheen_only(weight: f64, roughness: f64, tint: [f64; 3]) -> BsdfParams {
        let mut params = native([0.0, 0.0, 0.0, 1.0], 0.0);
        params.specular_ior_level = 0.0;
        params.sheen_weight = weight;
        params.sheen_roughness = roughness;
        params.sheen_tint = tint;
        params
    }

    #[test]
    fn single_bounce_sheen_r050_matches_blender() {
        assert_matches_oracle_at(
            "sheen_r050",
            &sheen_only(1.0, 0.5, [1.0; 3]),
            [0.178_175_499_328_062_87; 3],
            [5.081_926_241_668_168e-6; 3],
            FINE,
        );
    }

    #[test]
    fn single_bounce_sheen_tinted_r030_matches_blender() {
        assert_matches_oracle_at(
            "sheen_r030_tint",
            &sheen_only(0.6, 0.3, [1.0, 0.5, 0.25]),
            [
                4.087_217_123_924_347e-3,
                2.043_608_561_962_173_6e-3,
                1.021_804_280_981_086_8e-3,
            ],
            [
                2.675_749_113_963_379_7e-7,
                1.337_874_556_981_689_9e-7,
                6.689_372_784_908_449e-8,
            ],
            FINE,
        );
    }

    /// Sheen over a white base with the default specular layer: the sheen's layering attenuates the
    /// diffuse and the dielectric lobes, as Cycles layers it (`sheen_r050_base08`).
    #[test]
    fn single_bounce_sheen_over_white_base_matches_blender() {
        let mut params = native([0.8, 0.8, 0.8, 1.0], 0.0);
        params.roughness = 0.24;
        params.sheen_weight = 1.0;
        params.sheen_roughness = 0.5;
        params.sheen_tint = [1.0; 3];
        assert_matches_oracle_at(
            "sheen_r050_base08",
            &params,
            [6.065_270_029_474_05; 3],
            [7.655_433_604_088_337e-4; 3],
            // Measured on the sheen lobe alone at roughness 0.5 against a 1024² reference: COARSE is
            // 4.8e-5 off and FINE 5.0e-5. The test sums the diffuse and specular lobes too, which the
            // quadrature checks did not cover, so this is an assumption checked by the run.
            COARSE,
        );
    }
}

#[cfg(test)]
mod subsurface_tests {
    use super::*;
    use crate::model::Material;
    use crate::shader::SubsurfaceMethod;

    fn principled(method: SubsurfaceMethod, weight: f64) -> BsdfParams {
        let material = Material {
            base_color: [0.8, 0.5, 0.3, 1.0],
            roughness: 0.5,
            ..Material::default()
        };
        BsdfParams {
            subsurface_weight: weight,
            subsurface_method: method,
            subsurface_scale: 6.0,
            ..BsdfParams::from_material(&material, DVec3::Z)
        }
    }

    #[test]
    fn lobe_probabilities_sum_to_one_with_a_burley_subsurface_share() {
        let params = principled(SubsurfaceMethod::Burley, 0.5);
        let layers = layers(&params, DVec3::Z, DVec3::Z);
        let probabilities = lobe_probabilities(&params, &layers);
        let total = probabilities.diffuse
            + probabilities.subsurface
            + probabilities.sheen
            + probabilities.coat
            + probabilities.specular
            + probabilities.transmission;
        assert!((total - 1.0).abs() < 1.0e-12, "{total}");
        assert!(probabilities.subsurface > 0.0);
    }

    #[test]
    fn thin_wall_share_is_the_thin_branch_of_closure_setup() {
        // `svm/closure.h`: the diffuse keeps `1 - sw`, and `bsdf_thin_subsurface_setup` adds a reflection of weight
        // `sat(0.5 (1 - a)) × sw`. Both are Lambert, so one share. Anisotropy 0.5 gives a reflection of 0.25.
        let mut params = principled(SubsurfaceMethod::Burley, 0.5);
        params.thin_wall = true;
        params.subsurface_anisotropy = 0.5;
        let expected = 1.0 - 0.5 + 0.5 * 0.25;
        for channel in subsurface_share(&params).to_array() {
            assert!((channel - expected).abs() < 1e-12, "{channel}");
        }
    }

    #[test]
    fn smooth_thin_glass_has_shadow_transparency_but_rough_thin_glass_does_not() {
        let mut params = principled(SubsurfaceMethod::Burley, 0.0);
        params.transmission = 1.0;
        params.ior = 1.5;
        params.roughness = 0.0;
        params.base_color = [1.0; 4];
        params.glass_transmission_tint = [1.0; 3];
        params.thin_wall = true;
        let smooth = layers(&params, DVec3::Z, DVec3::Z);
        let transmittance = thin_glass_shadow_transparency(&params, &smooth);
        assert!(transmittance.x > 0.9, "{transmittance:?}");

        params.roughness = 0.2;
        let rough = layers(&params, DVec3::Z, DVec3::Z);
        assert_eq!(thin_glass_shadow_transparency(&params, &rough), DVec3::ZERO);
    }

    #[test]
    fn a_reflection_below_the_geometric_plane_is_lifted_and_a_valid_normal_is_kept() {
        // Cycles' `maybe_ensure_valid_specular_reflection`: the normal (0.9, 0, 0.436) about 64 degrees from Ng reflects
        // the view at 30 degrees to a direction below Ng (R_z ≈ -0.145), so it is raised until R_z reaches the
        // threshold min(0.9 cos_i, 0.01) = 0.01.
        let geometric = DVec3::Z;
        let view = DVec3::new(0.5, 0.0, 0.866_025_403_784_438_6);
        let tilted = DVec3::new(0.9, 0.0, 0.435_889_894_354_067_4).normalize();
        let reflect = |normal: DVec3| 2.0 * normal.dot(view) * normal - view;
        let threshold = (0.9 * view.dot(geometric)).min(0.01);
        assert!(geometric.dot(reflect(tilted)) < threshold);
        let bent = valid_specular_normal(geometric, tilted, view, true);
        assert!(
            geometric.dot(reflect(bent)) >= threshold - 1.0e-9,
            "the lifted reflection {} is below the threshold {threshold}",
            geometric.dot(reflect(bent))
        );
        assert!(
            bent.dot(geometric) > tilted.dot(geometric),
            "the normal rises towards Ng"
        );
        // A normal whose reflection is already above the threshold, and the geometric normal itself, are unchanged.
        assert_eq!(
            valid_specular_normal(geometric, DVec3::Z, view, true),
            DVec3::Z
        );
        assert_eq!(valid_specular_normal(geometric, view, view, true), view);
        // With `use_bump_map_correction` off, the normal is kept as it is (`maybe_ensure_valid_specular_reflection`).
        assert_eq!(
            valid_specular_normal(geometric, tilted, view, false),
            tilted
        );
    }

    #[test]
    fn the_subsurface_closure_is_zero_below_the_surface() {
        let params = principled(SubsurfaceMethod::Burley, 0.5);
        let layers = layers(&params, DVec3::Z, -DVec3::Z);
        assert_eq!(
            subsurface_closure_weight(
                &params,
                layers.energy_roughness,
                DVec3::Z,
                -DVec3::Z,
                layers.lower,
                true,
            ),
            DVec3::ZERO
        );
    }

    #[test]
    fn standalone_translucent_evaluates_and_samples_the_transmission_hemisphere() {
        let material = Material {
            base_color: [0.25, 0.5, 0.75, 1.0],
            ..Material::default()
        };
        let mut params = BsdfParams::from_material(&material, DVec3::Z);
        params.model = BsdfModel::Translucent;
        let layers = layers(&params, DVec3::Z, DVec3::Z);
        let light = -DVec3::Z;
        let color = DVec3::new(0.25, 0.5, 0.75);
        assert_eq!(standalone_eval(&params, &layers, light), color / PI);
        assert!((standalone_density(&params, &layers, light) - 1.0 / PI).abs() < 1.0e-15);
        let StandaloneSample::Continuous { direction, ray } =
            standalone_sample(&params, &layers, DVec3::Z, [0.5, 0.5, 0.0])
        else {
            panic!("the transmitted hemisphere sample should be continuous");
        };
        assert_eq!(direction, -DVec3::Z);
        assert_eq!(ray, RayKind::Transmission);
    }

    #[test]
    fn standalone_f82_selection_uses_the_cycles_z_half_lut_slices() {
        let scale = ggx_gen_schlick_s_f82(0.0, 0.0);
        assert!((scale - 0.999_496).abs() < 1.0e-7);
    }
}
