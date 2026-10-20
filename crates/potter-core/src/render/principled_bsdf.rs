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
//! Estimator contract: `evaluate` is the continuous BSDF times reflectance: the diffuse, specular,
//! sheen and coat lobes above the surface, and the glass lobe on both sides (its reflection above,
//! its transmission below). `continuous_density` is the density with which `path.rs` draws continuous
//! directions (lobe mixture), so `evaluate / density` is unbiased for the mixture over the sampler's
//! domain. The glass lobe's eval and density follow Cycles' `bsdf_microfacet_eval`, which also covers
//! half vectors the VNDF sampler never draws, so MIS on glass matches Cycles' eval/pdf and is not an
//! exact density. A singular glass lobe (roughness zero) is a delta that `sample_glass` draws; light
//! sampling cannot reach it.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use glam::DVec3;

use super::principled_tables::{
    ggx_energy, ggx_energy_average, ggx_gen_schlick_ior_s, ggx_glass_energy,
    ggx_glass_energy_average, ggx_glass_inverse_energy, ggx_glass_inverse_energy_average,
    sheen_ltc,
};
use super::subsurface::{burley_active, burley_radius, channel_mask};
use crate::shader::{BsdfParams, dielectric_f0};

/// Roughness floor shared with the sampler; Cycles treats roughness below ~1e-5 as delta.
const MIN_ROUGHNESS: f64 = 0.02;

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
    // The Burley subsurface closure takes its Subsurface Weight from the diffuse share (Cycles' diffuse weight is
    // `base × (1 - subsurface weight) × weight`, `svm/closure.h`).
    let subsurface_share = if burley_active(params) {
        params.subsurface_weight.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let diffuse_weight =
        ((1.0 - metallic) * (1.0 - transmission) * (1.0 - subsurface_share)).max(0.0);
    // The subsurface closure's sample weight is its mean channel weight times the channels it covers
    // (`bssrdf_setup`, `sample_weight = |average(weight)| * channels`).
    let covered_channels = channel_mask(burley_radius(params))
        .iter()
        .filter(|&&on| on)
        .count() as f64;
    let subsurface_weight =
        subsurface_closure_weight(params, layers.normal, layers.view, layers.lower).element_sum()
            / 3.0
            * covered_channels;
    // Each layer's selection weight is its layering albedo: Cycles' sample weight of the closure.
    let sheen_weight = layers.sheen.selection_weight();
    let coat_weight = layers.coat.map_or(0.0, |coat| coat.albedo);
    // The glass closure is weighted `(1 - metallic) · transmission`: Cycles takes the metallic weight
    // first (`kernel/svm/closure.h`). A view below the surface has no glass lobe.
    let glass_weight = if layers.glass.is_some() {
        (1.0 - metallic) * transmission
    } else {
        0.0
    };
    // The dielectric specular layer is weighted `(1 - metallic) · (1 - transmission)`, as `specular_lobe`
    // evaluates it.
    let specular_weight = (metallic
        + dielectric_f0(params.ior, params.specular_ior_level)
            * (1.0 - metallic)
            * (1.0 - transmission))
        .max(1.0e-4);
    let total = diffuse_weight
        + subsurface_weight
        + sheen_weight
        + coat_weight
        + glass_weight
        + specular_weight;
    LobeProbabilities {
        diffuse: diffuse_weight / total,
        subsurface: subsurface_weight / total,
        sheen: sheen_weight / total,
        coat: coat_weight / total,
        specular: specular_weight / total,
        transmission: glass_weight / total,
    }
}

/// Density of the continuous (diffuse, sheen, coat and specular) directions drawn by `path.rs`, in solid
/// angle, with the glass lobe's density as Cycles' eval and pdf define it. Diffuse is cosine-distributed;
/// sheen is the LTC distribution; the coat and specular lobes are the GGX half-vector distribution D·cosθh
/// mapped to the reflected direction. The glass lobe's density is its reflection above the surface and its
/// refraction below it.
pub(crate) fn continuous_density(params: &BsdfParams, layers: &Layers, light: DVec3) -> f64 {
    let (normal, view) = (layers.normal, layers.view);
    let probabilities = lobe_probabilities(params, layers);
    let mut density = probabilities.diffuse * normal.dot(light).max(0.0) / PI;
    density += probabilities.sheen * layers.sheen.density(light);
    if let Some(coat) = layers.coat {
        density += probabilities.coat * half_vector_pdf(coat.normal, coat.roughness, view, light);
    }
    density += probabilities.specular * half_vector_pdf(normal, params.roughness, view, light);
    if let Some(glass) = &layers.glass {
        density += probabilities.transmission * glass_eval(glass, normal, view, light).1;
    }
    density
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

/// Diffuse lobe of the Principled BSDF: Lambert under the dielectric-layering weight, times reflectance, less the
/// share the subsurface closure takes. The raster preview has no subsurface closure, so it shades the lobe without
/// that share (the realtime gap); the path tracer uses `evaluate`, which keeps it.
pub(crate) fn diffuse_coefficient(params: &BsdfParams, layers: &Layers) -> DVec3 {
    let without_subsurface = BsdfParams {
        subsurface_weight: 0.0,
        ..*params
    };
    diffuse_lobe(
        &without_subsurface,
        layers.normal,
        layers.view,
        layers.lower,
    )
}

/// The Lambert weight of the dielectric-layered diffuse lobe, before the 1/π and the albedo (Cycles' diffuse weight).
fn diffuse_layer_weight(params: &BsdfParams, normal: DVec3, view: DVec3) -> f64 {
    let n_dot_v = normal.dot(view).max(0.0);
    let roughness = params.roughness.clamp(MIN_ROUGHNESS, 1.0);
    let metallic = params.metallic.clamp(0.0, 1.0);
    let transmission = params.transmission.clamp(0.0, 1.0);
    let dielectric_weight = ((1.0 - metallic) * (1.0 - transmission)).max(0.0);
    match specular_layer(params) {
        Some(layer) => {
            let z = ((layer.eta - 1.0) / (layer.eta + 1.0)).abs().sqrt();
            let albedo_remap = ggx_gen_schlick_ior_s(roughness, n_dot_v, z);
            let albedo = layer.f0_tinted.lerp(DVec3::ONE, albedo_remap);
            let energy = ggx_energy(roughness, n_dot_v).max(1.0e-6);
            let darkened = albedo * dielectric_darkening(&layer, roughness, energy);
            layered_diffuse_weight(dielectric_weight, darkened.max_element())
        }
        None => dielectric_weight,
    }
}

/// The per-channel share of the diffuse base that the Burley subsurface closure leaves to diffuse: `1 - Subsurface
/// Weight` on the channels with a radius, and `1` on the others, which fall back to diffuse (`bssrdf_setup`).
fn subsurface_share(params: &BsdfParams) -> DVec3 {
    if !burley_active(params) {
        return DVec3::ONE;
    }
    let sss = params.subsurface_weight.clamp(0.0, 1.0);
    let mask = channel_mask(burley_radius(params));
    DVec3::from_array(std::array::from_fn(|channel| {
        if mask[channel] { 1.0 - sss } else { 1.0 }
    }))
}

/// The Burley subsurface closure's weight per channel, `clamped base × subsurface weight × diffuse layer weight ×
/// reflectance × lower`, zero on channels without a radius (Cycles' `closure_weight` of the BSSRDF, `svm/closure.h`).
/// It is the weight the path tracer divides by the branch's probability to enter the surface.
pub(crate) fn subsurface_closure_weight(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    lower: DVec3,
) -> DVec3 {
    if !burley_active(params) || normal.dot(view) <= 0.0 {
        return DVec3::ZERO;
    }
    let mask = channel_mask(burley_radius(params));
    let covered = DVec3::from_array(mask.map(|on| if on { 1.0 } else { 0.0 }));
    base_color(params)
        * covered
        * (params.subsurface_weight.clamp(0.0, 1.0)
            * diffuse_layer_weight(params, normal, view)
            * params.reflectance.clamp(0.0, 1.0))
        * lower
}

/// The diffuse lobe, given the coat's lower-layer factor (set up once by the caller).
fn diffuse_lobe(params: &BsdfParams, normal: DVec3, view: DVec3, lower: DVec3) -> DVec3 {
    let n_dot_v = normal.dot(view).max(0.0);
    if n_dot_v <= 0.0 {
        return DVec3::ZERO;
    }
    let weight = diffuse_layer_weight(params, normal, view);
    diffuse_albedo(params)
        * subsurface_share(params)
        * (weight / PI)
        * params.reflectance.clamp(0.0, 1.0)
        * lower
}

/// Specular (dielectric and metallic) lobes along `light`, times reflectance. Zero when the
/// light or view is below the surface.
pub(crate) fn specular_coefficient(params: &BsdfParams, layers: &Layers, light: DVec3) -> DVec3 {
    specular_lobe(params, layers.normal, layers.view, light, layers.lower)
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
        let real_f = fresnel_dielectric(v_dot_h, layer.eta);
        let fresnel = remapped_fresnel(f0, layer.eta, real_f);
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
pub(crate) fn evaluate(params: &BsdfParams, layers: &Layers, light: DVec3) -> DVec3 {
    // The reflective lobes sit above the surface; the glass lobe takes both sides.
    let reflective = if layers.normal.dot(light) > 0.0 {
        diffuse_lobe(params, layers.normal, layers.view, layers.lower)
            + specular_lobe(params, layers.normal, layers.view, light, layers.lower)
            + sheen_coefficient(params, layers, light)
            + coat_coefficient(params, layers, light)
    } else {
        DVec3::ZERO
    };
    reflective + glass_coefficient(params, layers, light)
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

/// The layers above the base lobes at one shading point, set up once and shared by every lobe they
/// attenuate. `lower` is the factor they leave to the diffuse, specular and metallic lobes, the
/// transmission delta and the emission: the coat's layering and tint, times the sheen's layering,
/// as Cycles layers them (`principled_bsdf_emission`, `kernel/svm/closure.h`).
#[derive(Clone, Copy)]
pub(crate) struct Layers {
    pub(crate) normal: DVec3,
    pub(crate) view: DVec3,
    coat: Option<CoatLayer>,
    sheen: SheenLobe,
    pub(crate) lower: DVec3,
    glass: Option<GlassLobe>,
}

/// Sets up the sheen and coat layers of a shading point with normal `normal`, seen along `view`.
pub(crate) fn layers(params: &BsdfParams, normal: DVec3, view: DVec3) -> Layers {
    let coat = coat_layer(params, view);
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
    let glass = glass_lobe(params, normal, view);
    Layers {
        normal,
        view,
        coat,
        sheen,
        lower,
        glass,
    }
}

impl Layers {
    /// The factor the layers leave to the lobes below them and to the emission.
    pub(crate) fn lower(&self) -> DVec3 {
        self.lower
    }

    /// Sheen direction for the square sample `(u, v)` in `[0, 1)²`, drawn from the LTC distribution.
    pub(crate) fn sample_sheen(&self, u: f64, v: f64) -> DVec3 {
        self.sheen.direction(u, v)
    }

    /// Whether a light below the surface can reach it through the glass lobe. A singular lobe is drawn only as a
    /// delta, which light sampling cannot reach, so it does not count.
    pub(crate) fn transmits(&self) -> bool {
        self.glass.is_some_and(|glass| !glass.singular)
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
fn concentric_disk(u: f64, v: f64) -> (f64, f64) {
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

/// Cycles' `make_orthonormals` (`util/math_float3.h`): a tangent from the cross product with
/// `(1, 1, 1)`, chosen by exact component equality.
#[expect(
    clippy::float_cmp,
    reason = "Cycles selects the tangent basis by exact component equality of the normal"
)]
pub(super) fn make_orthonormals(normal: DVec3) -> (DVec3, DVec3) {
    let tangent = if normal.x != normal.y || normal.x != normal.z {
        DVec3::new(
            normal.z - normal.y,
            normal.x - normal.z,
            normal.y - normal.x,
        )
    } else {
        DVec3::new(
            normal.z - normal.y,
            normal.x + normal.z,
            -normal.y - normal.x,
        )
    }
    .normalize_or_zero();
    (tangent, normal.cross(tangent))
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

/// Cycles' GGX glass closure: one closure with a reflection lobe and a transmission lobe, set up as
/// `bsdf_microfacet_ggx_glass_setup` and `generalized_schlick_setup` (`kernel/closure/bsdf_microfacet.h`),
/// with the multiscatter energy compensation of `microfacet_ggx_preserve_energy`. The Principled BSDF's
/// Transmission Weight (`kernel/svm/closure.h`) and the Glass BSDF node both build it.
#[derive(Clone, Copy, Debug)]
struct GlassLobe {
    /// Cycles' `bsdf->ior`: the closure's IOR, inverted at a back face.
    ior: f64,
    /// GGX `alpha`, the roughness squared (Cycles' `alpha_x`).
    alpha: f64,
    /// Reflectance at normal incidence, tinted and saturated (`generalized_schlick_setup`'s `f0`).
    f0: DVec3,
    reflection_tint: DVec3,
    transmission_tint: DVec3,
    /// Multiscatter gain `1 + Fms · (1 - E) / E` per channel, from the view. It scales the eval and not
    /// the density, as Cycles' `energy_scale` and the darkening of the weight together do.
    gain: DVec3,
    /// Roughness below Cycles' almost-specular threshold: the eval is zero and `sample_glass` draws a delta.
    singular: bool,
    /// An IOR of 1. Cycles' `bsdf_microfacet_sample` makes a refraction there a delta (the straight-through
    /// direction), while `bsdf_microfacet_eval` keeps the roughness-only flag, so the eval stays rough.
    straight: bool,
}

/// The glass lobe at a shading point with normal `normal` seen along `view`, or `None` when its weight
/// `(1 - metallic) · transmission` is zero or the view is below the normal.
fn glass_lobe(params: &BsdfParams, normal: DVec3, view: DVec3) -> Option<GlassLobe> {
    let metallic = params.metallic.clamp(0.0, 1.0);
    let transmission = params.transmission.clamp(0.0, 1.0);
    let cos_ni = normal.dot(view);
    if (1.0 - metallic) * transmission <= 0.0 || cos_ni <= 0.0 {
        return None;
    }
    let roughness = params.roughness.clamp(0.0, 1.0);
    let alpha = roughness * roughness;
    let input_ior = params.ior.max(1.0e-5);
    let ior = if params.back_facing {
        1.0 / input_ior
    } else {
        input_ior
    };
    // `generalized_schlick_setup` tints F0 of the closure's own IOR, which a back face does not invert.
    let f0 = (DVec3::from_array(params.specular_tint).max(DVec3::ZERO)
        * dielectric_f0(input_ior, 0.5))
    .clamp(DVec3::ZERO, DVec3::ONE);
    let reflection_tint = DVec3::from_array(params.glass_reflection_tint).max(DVec3::ZERO);
    let transmission_tint = DVec3::from_array(params.glass_transmission_tint).max(DVec3::ZERO);
    let singular = alpha * alpha <= GLASS_SINGULAR_ALPHA_SQUARED;
    let straight = (ior - 1.0).abs() < 1.0e-4;

    // `microfacet_ggx_preserve_energy`: the GGX_GLASS tables at the view, the inverse ones below an IOR of 1.
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
    let fss = if transmission_tint.max_element() <= 0.0 {
        // The reflective Fss: the dielectric's average single-scattering Fresnel, on the reflection tint.
        dielectric_fss(&SpecularLayer {
            eta: ior,
            f0_tinted: f0,
        }) * reflection_tint
    } else {
        transmission_tint
    };
    let gain = multiscatter_gain(fss, energy_average, missing);
    Some(GlassLobe {
        ior,
        alpha,
        f0,
        reflection_tint,
        transmission_tint,
        gain,
        singular,
        straight,
    })
}

impl GlassLobe {
    /// Cycles' `generalized_schlick_fresnel` with the real dielectric remap: the reflectance and the
    /// transmittance at the cosine `cos_hi` between the half vector and the view, and the cosine of the
    /// refracted angle.
    fn fresnel(&self, cos_hi: f64) -> (DVec3, DVec3, f64) {
        let (real, cos_ht) = dielectric_fresnel(cos_hi, self.ior);
        let fresnel = remapped_fresnel(self.f0, self.ior, real);
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

/// The glass lobe along `light`, times reflectance and the layers above it: its reflection above the
/// surface and its transmission below. A delta lobe is zero here, since `sample_glass` draws it.
pub(crate) fn glass_coefficient(params: &BsdfParams, layers: &Layers, light: DVec3) -> DVec3 {
    let Some(lobe) = layers.glass.as_ref() else {
        return DVec3::ZERO;
    };
    let cosine = layers.normal.dot(light).abs();
    if cosine <= 0.0 {
        return DVec3::ZERO;
    }
    let (value, _) = glass_eval(lobe, layers.normal, layers.view, light);
    value / cosine * (params.reflectance.clamp(0.0, 1.0) * layers.lower)
}

/// A draw from the glass closure, as Cycles' `bsdf_microfacet_sample` makes it.
pub(crate) enum GlassSample {
    /// A delta reflection or refraction. `weight` is the Fresnel share over the branch probability (`branch`,
    /// the glass probability in the lobe mixture), times the energy gain, the reflectance and the layers.
    Singular { direction: DVec3, weight: DVec3 },
    /// A rough draw, which the caller weights by the lobe mixture's density.
    Rough { direction: DVec3 },
    /// A draw on the wrong side of the normal or the geometric normal: Cycles' `LABEL_NONE`.
    Absorbed,
}

/// Draws the glass closure from three uniform numbers: a visible-normal half vector (the normal for a
/// delta lobe), then reflection or refraction, refraction with probability `1 - reflectance`.
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
    let (normal, view) = (layers.normal, layers.view);
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
    let scale = params.reflectance.clamp(0.0, 1.0) * layers.lower;
    GlassSample::Singular {
        direction,
        weight: value * lobe.gain * scale / (branch * share),
    }
}

/// Cycles' `microfacet_ggx_sample_vndf` (Heitz 2018): the half vector visible from the local view `wi`
/// (`z` up) for the isotropic GGX `alpha`, from the uniform pair `(u, v)`.
fn sample_visible_normal(wi: DVec3, alpha: f64, u: f64, v: f64) -> DVec3 {
    let stretched = DVec3::new(alpha * wi.x, alpha * wi.y, wi.z).normalize_or_zero();
    let length_squared = stretched.x * stretched.x + stretched.y * stretched.y;
    let (t1, t2) = if length_squared > 1.0e-7 {
        let t1 = DVec3::new(-stretched.y, stretched.x, 0.0) / length_squared.sqrt();
        (t1, stretched.cross(t1))
    } else {
        (DVec3::X, DVec3::Y)
    };
    // Section 4.2: the parameterisation of the projected area.
    let (x, disk_y) = concentric_disk(u, v);
    let s = f64::midpoint(1.0, stretched.z);
    let y = (1.0 - s) * (1.0 - x * x).max(0.0).sqrt() + s * disk_y;
    // Section 4.3: the reprojection onto the hemisphere, then back to the ellipsoid.
    let hemisphere = DVec3::new(x, y, (1.0 - x * x - y * y).max(0.0).sqrt());
    let half = t1 * hemisphere.x + t2 * hemisphere.y + stretched * hemisphere.z;
    DVec3::new(alpha * half.x, alpha * half.y, half.z.max(0.0)).normalize_or_zero()
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
        let Some(lobe) = glass_lobe(&params, DVec3::Z, view()) else {
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
        let Some(lobe) = glass_lobe(&params, DVec3::Z, view) else {
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
            ior,
            alpha: 0.25,
            f0: DVec3::splat(dielectric_f0(ior, 0.5)),
            reflection_tint: DVec3::ONE,
            transmission_tint: DVec3::ONE,
            gain: DVec3::ONE,
            singular: false,
            straight: false,
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
            let GlassSample::Singular { direction, weight } =
                sample_glass(&params, &shading, DVec3::Z, 1.0, [0.5, 0.5, u])
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
            let GlassSample::Singular { direction, weight } =
                sample_glass(&params, &shading, DVec3::Z, 1.0, [0.3, 0.7, u])
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
                total += evaluate(params, &layers(params, normal, view), direction) * weight;
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
    fn an_unevaluated_method_has_no_subsurface_lobe() {
        // A method Potter does not evaluate must not take probability mass from the diffuse lobe.
        let params = principled(SubsurfaceMethod::Mixed, 0.5);
        let layers = layers(&params, DVec3::Z, DVec3::Z);
        assert_eq!(lobe_probabilities(&params, &layers).subsurface, 0.0);
        assert_eq!(
            subsurface_closure_weight(&params, DVec3::Z, DVec3::Z, layers.lower),
            DVec3::ZERO
        );
    }

    #[test]
    fn the_subsurface_closure_is_zero_below_the_surface() {
        let params = principled(SubsurfaceMethod::Burley, 0.5);
        let layers = layers(&params, DVec3::Z, -DVec3::Z);
        assert_eq!(
            subsurface_closure_weight(&params, DVec3::Z, -DVec3::Z, layers.lower),
            DVec3::ZERO
        );
    }
}
