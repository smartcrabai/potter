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
//! - `kernel/closure/bsdf_sheen.h`: the microfiber sheen lobe (`bsdf_sheen_setup`, `_eval`,
//!   `_sample`) over `table_sheen_ltc`. Its frame is `make_orthonormals_safe_tangent(N, wi)`
//!   (`kernel/sample/mapping.h`). It is set up first, so its layering weight attenuates every lobe
//!   below it (`svm/closure.h`, `principled_bsdf_emission`).
//! - `node_shader_bsdf_principled.cc`: the default distribution is `SHD_GLOSSY_MULTI_GGX`, so
//!   energy preservation is enabled for the specular and metallic lobes.
//!
//! Estimator contract: `evaluate` is the continuous BSDF (diffuse, sheen and specular lobes,
//! reflectance applied). `continuous_density` is the exact density with which `path.rs` draws
//! continuous directions (lobe mixture), so `evaluate / density` is unbiased for the mixture. The
//! transmission lobe stays a delta, handled by `path.rs`.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use glam::DVec3;

use super::principled_tables::{ggx_energy, ggx_energy_average, ggx_gen_schlick_ior_s, sheen_ltc};
use crate::shader::{BsdfParams, dielectric_f0};

/// Roughness floor shared with the sampler; Cycles treats roughness below ~1e-5 as delta.
const MIN_ROUGHNESS: f64 = 0.02;

/// Cycles' `CLOSURE_WEIGHT_CUTOFF`: a closure weight at or below it is not set up at all.
const CLOSURE_WEIGHT_CUTOFF: f64 = 1.0e-5;

/// Selection probabilities of the four sampling branches, identical to the thresholds `path.rs`
/// uses to pick a branch: transmission, diffuse, sheen, specular.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LobeProbabilities {
    pub(crate) diffuse: f64,
    pub(crate) sheen: f64,
    pub(crate) specular: f64,
    pub(crate) transmission: f64,
}

pub(crate) fn lobe_probabilities(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
) -> LobeProbabilities {
    let metallic = params.metallic.clamp(0.0, 1.0);
    let transmission = params.transmission.clamp(0.0, 1.0);
    let diffuse_weight = ((1.0 - metallic) * (1.0 - transmission)).max(0.0);
    let sheen_weight = sheen_lobe(params, normal, view).selection_weight();
    let specular_weight = (metallic
        + dielectric_f0(params.ior, params.specular_ior_level) * (1.0 - metallic))
        .max(1.0e-4);
    let total = diffuse_weight + sheen_weight + transmission + specular_weight;
    LobeProbabilities {
        diffuse: diffuse_weight / total,
        sheen: sheen_weight / total,
        specular: specular_weight / total,
        transmission: transmission / total,
    }
}

/// Exact density of the continuous (diffuse, sheen and specular) directions drawn by `path.rs`, in
/// solid angle. Diffuse is cosine-distributed; sheen is the LTC distribution; specular is the GGX
/// half-vector distribution D·cosθh mapped to the reflected direction.
pub(crate) fn continuous_density(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    light: DVec3,
) -> f64 {
    let probabilities = lobe_probabilities(params, normal, view);
    let n_dot_l = normal.dot(light).max(0.0);
    let mut density = probabilities.diffuse * n_dot_l / PI;
    density += probabilities.sheen * sheen_lobe(params, normal, view).density(light);
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
            let energy = ggx_energy(roughness, n_dot_v).max(1.0e-6);
            let darkened = albedo * dielectric_darkening(&layer, roughness, energy);
            layered_diffuse_weight(dielectric_weight, darkened.max_element())
        }
        None => dielectric_weight,
    };
    let sheen_layer = sheen_lobe(params, normal, view).layer_weight();
    diffuse_albedo(params)
        * (diffuse_weight * sheen_layer / PI)
        * params.reflectance.clamp(0.0, 1.0)
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

    let sheen_layer = sheen_lobe(params, normal, view).layer_weight();
    (reflected * common) * (params.reflectance.clamp(0.0, 1.0) * sheen_layer)
}

/// Continuous Principled BSDF times reflectance, the `evaluate / density` estimator's numerator.
pub(crate) fn evaluate(params: &BsdfParams, normal: DVec3, view: DVec3, light: DVec3) -> DVec3 {
    if normal.dot(light) <= 0.0 {
        return DVec3::ZERO;
    }
    diffuse_coefficient(params, normal, view)
        + specular_coefficient(params, normal, view, light)
        + sheen_coefficient(params, normal, view, light)
}

/// The sheen lobe along `light`, times reflectance. It depends on the light, so an area light is
/// integrated numerically or at its centre (`scene.rs`), like the specular lobe.
pub(crate) fn sheen_coefficient(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    light: DVec3,
) -> DVec3 {
    sheen_lobe(params, normal, view).value(light) * params.reflectance.clamp(0.0, 1.0)
}

/// The reflectance the sheen layer leaves to the lobes below it (Cycles' layering weight of the
/// sheen closure). `path.rs` scales the transmission lobe by it.
pub(crate) fn sheen_layer_weight(params: &BsdfParams, normal: DVec3, view: DVec3) -> f64 {
    sheen_lobe(params, normal, view).layer_weight()
}

/// Sheen direction for the square sample `(u, v)` in `[0, 1)²`, drawn from the LTC distribution.
/// `path.rs` supplies the sample, as it does for the other lobes.
pub(crate) fn sample_sheen(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    u: f64,
    v: f64,
) -> DVec3 {
    sheen_lobe(params, normal, view).direction(u, v)
}

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
fn make_orthonormals(normal: DVec3) -> (DVec3, DVec3) {
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
        let material = Material {
            base_color: [0.0, 0.0, 0.0, 1.0],
            metallic: 0.0,
            roughness: 0.24,
            ..Material::default()
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
        let view = DVec3::new(0.5, 0.0, 0.5_f64.sqrt());
        let params = sheen_params(1.0, 0.5, [1.0; 3]);
        const SIDE: usize = 400;
        for cap in [0.6_f64, 1.0] {
            let inside = (0..SIDE)
                .flat_map(|i| (0..SIDE).map(move |j| (i, j)))
                .filter(|&(i, j)| {
                    let u = (i as f64 + 0.5) / SIDE as f64;
                    let v = (j as f64 + 0.5) / SIDE as f64;
                    let direction = sample_sheen(&params, DVec3::Z, view, u, v);
                    direction.z >= cap.cos()
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
        let layer = sheen_layer_weight(&params, DVec3::Z, view);
        assert!((layer - (1.0 - 0.6 * albedo)).abs() < 1.0e-12);
        let off = sheen_params(0.0, 0.3, [1.0; 3]);
        assert_eq!(sheen_layer_weight(&off, DVec3::Z, view), 1.0);
        assert_eq!(sheen_coefficient(&off, DVec3::Z, view, view), DVec3::ZERO);
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
    /// Quadrature resolution over the disk's bounding square. 512 is needed at this radius: 256 is
    /// 0.02% off on the diffuse variant, more than its Blender stderr.
    const GRID: usize = 512;
    /// Orthographic width of the probe camera (Blender's `ortho_scale`) in units.
    const ORTHO_SCALE: f64 = 0.2;

    /// Mean over the central 8×8 pixel patch of the orthographic probe (16 pixels across 0.2 units).
    /// The camera views the plane at `VIEW_THETA`, so the image's vertical axis maps to receiver x
    /// with a factor `cos(VIEW_THETA)`: one pixel is `pixel / cos` in x and `pixel` in y. A glossy
    /// highlight depends on the receiver position, so the patch is averaged, as Blender's pixel is.
    /// The probe is symmetric under y -> -y, so rows j and 7-j contribute equally.
    fn probe_pixel(params: &BsdfParams) -> DVec3 {
        let pixel = ORTHO_SCALE / 16.0;
        let x_step = pixel / VIEW_THETA.cos();
        let mut total = DVec3::ZERO;
        for i in 0..8 {
            for j in 0..4 {
                let x = (f64::from(i) - 3.5) * x_step;
                let y = (f64::from(j) - 3.5) * pixel;
                total += 2.0 * receiver_integral(params, x, y);
            }
        }
        total / 64.0
    }

    /// Single-bounce integral at the receiver `(x, y, 0)`.
    fn receiver_integral(params: &BsdfParams, x: f64, y: f64) -> DVec3 {
        let normal = DVec3::Z;
        let view = DVec3::new(VIEW_THETA.sin(), 0.0, VIEW_THETA.cos());
        let centre = LIGHT_DISTANCE * DVec3::new(-VIEW_THETA.sin(), 0.0, VIEW_THETA.cos());
        let toward_receiver = -centre.normalize();
        let u = toward_receiver.cross(DVec3::Y).normalize();
        let w = toward_receiver.cross(u);
        let receiver = DVec3::new(x, y, 0.0);
        let radiance = POWER / (std::f64::consts::PI * std::f64::consts::PI * RADIUS * RADIUS);
        let h = 2.0 / GRID as f64;
        let mut total = DVec3::ZERO;
        for i in 0..GRID {
            for j in 0..GRID {
                let a = (i as f64 + 0.5) * h - 1.0;
                let b = (j as f64 + 0.5) * h - 1.0;
                if a * a + b * b > 1.0 {
                    continue;
                }
                let point = centre + RADIUS * (a * u + b * w);
                let offset = point - receiver;
                let distance = offset.length();
                let direction = offset / distance;
                let cos_receiver = direction.dot(normal).max(0.0);
                let cos_emitter = (-direction).dot(toward_receiver).max(0.0);
                let weight = RADIUS * RADIUS * h * h * cos_emitter / (distance * distance)
                    * radiance
                    * cos_receiver;
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
        let potter = probe_pixel(params);
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

    /// Sheen alone: the base diffuse is black and Specular IOR Level 0 removes the dielectric layer
    /// (`PROBE_DIAMETER=2.4 PROBE_VARIANTS=sheen_r050,sheen_r030_tint`, the committed light size).
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
        assert_matches_oracle(
            "sheen_r050",
            &sheen_only(1.0, 0.5, [1.0; 3]),
            [0.178_175_499_328_062_87; 3],
            [5.081_926_241_668_168e-6; 3],
        );
    }

    #[test]
    fn single_bounce_sheen_tinted_r030_matches_blender() {
        assert_matches_oracle(
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
        );
    }
}
