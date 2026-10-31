//! Hair BSDF kernels ported from Cycles v5.2.2.
//!
//! Principled Huang setup consumes the real curve-normal attribute and camera pixel coverage from
//! [`HairHitData`]. Its evaluation and sampling also advance the caller's Cycles LCG state.

use std::f64::consts::{FRAC_PI_2, PI};

use glam::DVec3;

use crate::{
    model::RayKind,
    shader::{
        BsdfModel, BsdfParams, HairComponent, HairParametrization, HairScatteringModel,
        make_orthonormals,
    },
};

/// Hit-only data used by the Principled Hair closure. `ribbon_v` carries Cycles' ribbon coordinate
/// for ribbon hits; thick curves use the normal-derived cross-section. `curve_normal` is the
/// interpolated Cycles `attr_normal` used to orient an elliptical cross-section. `pixel_coverage` is
/// `0.5 * sd->dP / interpolated_curve_radius` for camera hits, and 1 for non-camera hits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HairHitData {
    /// Curve-normal attribute for Huang's ellipse; source point values are interpolated without normalization.
    pub(crate) curve_normal: DVec3,
    pub(crate) pixel_coverage: f64,
    pub(crate) ribbon_v: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum HairSample {
    /// Cycles returns the sampled value and density together; the path must not re-evaluate this hair closure.
    Continuous {
        direction: DVec3,
        ray: RayKind,
        eval: DVec3,
        pdf: f64,
    },
    Absorbed,
}

const M_1_2PI: f64 = 1.0 / (2.0 * PI);
const M_2_PI: f64 = 2.0 / PI;
const REC709_LUMA: DVec3 = DVec3::new(0.2126, 0.7152, 0.0722);
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

fn exact_f64_one(value: f64) -> bool {
    value.to_bits() == 1.0_f64.to_bits()
}

/// Hair evaluation is consumed directly by Cycles' path integrator, without an extra surface cosine.
pub(crate) fn hair_eval(
    params: &BsdfParams,
    incoming: DVec3,
    outgoing: DVec3,
    tangent: DVec3,
    radial_normal: DVec3,
    hit: HairHitData,
    rng: &mut u32,
) -> DVec3 {
    match params.model {
        BsdfModel::Hair(hair) => {
            let (value, _) = classic_hair_eval(hair, incoming, outgoing, radial_normal);
            value * closure_color(params)
        }
        BsdfModel::PrincipledHair(hair) => {
            let sigma = principled_sigma(params, hair);
            match hair.model {
                HairScatteringModel::Chiang => {
                    let setup = chiang_setup(hair, sigma);
                    let (value, _) =
                        chiang_eval(setup, incoming, outgoing, tangent, radial_normal, hit);
                    value
                }
                HairScatteringModel::Huang => {
                    let setup = huang_setup(hair, sigma, incoming, tangent, radial_normal, hit);
                    huang_eval(setup, outgoing, rng)
                }
            }
        }
        _ => unreachable!("hair kernel called for a non-hair BSDF"),
    }
}

/// Solid-angle density of the matching Cycles closure sampler.
pub(crate) fn hair_pdf(
    params: &BsdfParams,
    incoming: DVec3,
    outgoing: DVec3,
    tangent: DVec3,
    radial_normal: DVec3,
    hit: HairHitData,
) -> f64 {
    match params.model {
        BsdfModel::Hair(hair) => classic_hair_eval(hair, incoming, outgoing, radial_normal).1,
        BsdfModel::PrincipledHair(hair) => match hair.model {
            HairScatteringModel::Chiang => {
                let setup = chiang_setup(hair, principled_sigma(params, hair));
                chiang_eval(setup, incoming, outgoing, tangent, radial_normal, hit).1
            }
            // Cycles sets Huang's PDF to one before all early-outs and uses the same value for
            // emitter and BSDF sampling; its stochastic residual is carried in the value.
            HairScatteringModel::Huang => 1.0,
        },
        _ => unreachable!("hair kernel called for a non-hair BSDF"),
    }
}

/// Whether Cycles' Huang setup replaces this hair BSDF with a transparent closure because the
/// hit lies outside the projected ellipse.
pub(crate) fn hair_is_transparent(
    params: &BsdfParams,
    incoming: DVec3,
    tangent: DVec3,
    radial_normal: DVec3,
    hit: HairHitData,
) -> bool {
    let BsdfModel::PrincipledHair(hair) = params.model else {
        return false;
    };
    if hair.model != HairScatteringModel::Huang {
        return false;
    }
    let setup = huang_setup(
        hair,
        principled_sigma(params, hair),
        incoming,
        tangent,
        radial_normal,
        hit,
    );
    setup.h.abs() >= setup.radius
}

pub(crate) fn hair_sample(
    params: &BsdfParams,
    incoming: DVec3,
    tangent: DVec3,
    radial_normal: DVec3,
    hit: HairHitData,
    random: [f64; 4],
    rng: &mut u32,
) -> HairSample {
    match params.model {
        BsdfModel::Hair(hair) => classic_hair_sample(hair, incoming, closure_color(params), random),
        BsdfModel::PrincipledHair(hair) => match hair.model {
            HairScatteringModel::Chiang => chiang_sample(
                chiang_setup(hair, principled_sigma(params, hair)),
                incoming,
                tangent,
                radial_normal,
                hit,
                random,
            ),
            HairScatteringModel::Huang => {
                let setup = huang_setup(
                    hair,
                    principled_sigma(params, hair),
                    incoming,
                    tangent,
                    radial_normal,
                    hit,
                );
                huang_sample(setup, random, rng)
            }
        },
        _ => unreachable!("hair kernel called for a non-hair BSDF"),
    }
}

fn closure_color(params: &BsdfParams) -> DVec3 {
    DVec3::new(
        params.base_color[0],
        params.base_color[1],
        params.base_color[2],
    )
    .max(DVec3::ZERO)
}

pub(crate) fn hair_sample_weight(params: &BsdfParams) -> f64 {
    match params.model {
        BsdfModel::Hair(_) => {
            let color = closure_color(params);
            let weight = (color.x + color.y + color.z) / 3.0;
            if weight >= 1.0e-5 { weight.abs() } else { 0.0 }
        }
        BsdfModel::PrincipledHair(hair) => match hair.model {
            HairScatteringModel::Chiang => 1.0,
            HairScatteringModel::Huang => {
                if hair.reflection > 0.0
                    || hair.transmission > 0.0
                    || hair.secondary_reflection > 0.0
                {
                    1.0
                } else {
                    0.0
                }
            }
        },
        _ => unreachable!("hair weight requested for a non-hair BSDF"),
    }
}

fn principled_sigma(params: &BsdfParams, hair: crate::shader::PrincipledHairParameters) -> DVec3 {
    let random_factor = random_roughness_factor(hair.random, hair.random_roughness);
    let radial_roughness = match hair.model {
        HairScatteringModel::Chiang => hair.radial_roughness * random_factor,
        HairScatteringModel::Huang => hair.roughness * random_factor,
    };
    match hair.parametrization {
        HairParametrization::Reflectance => {
            sigma_from_reflectance(closure_color(params), radial_roughness)
        }
        HairParametrization::DirectAbsorption => DVec3::from_array(hair.absorption_coefficient),
        HairParametrization::MelaninConcentration => {
            let random_color = hair.random_color.clamp(0.0, 1.0);
            let factor_random_color = 1.0 + 2.0 * (hair.random - 0.5) * random_color;
            let melanin = hair.melanin * factor_random_color;
            let melanin = -(1.0 - melanin).max(0.0001).ln();
            let eumelanin = melanin * (1.0 - hair.melanin_redness);
            let pheomelanin = melanin * hair.melanin_redness;
            let melanin_sigma = DVec3::new(0.506, 0.841, 1.653) * eumelanin
                + DVec3::new(0.343, 0.733, 1.924) * pheomelanin;
            melanin_sigma + sigma_from_reflectance(DVec3::from_array(hair.tint), radial_roughness)
        }
    }
}

fn sigma_from_reflectance(color: DVec3, azimuthal_roughness: f64) -> DVec3 {
    let x = azimuthal_roughness;
    let scale = (((((0.245 * x) + 5.574) * x - 10.73) * x + 2.532) * x - 0.215) * x + 5.969;
    let log_color = color.max(DVec3::ZERO).ln() / scale;
    log_color * log_color
}

fn random_roughness_factor(random: f64, amount: f64) -> f64 {
    1.0 + 2.0 * (random - 0.5) * amount
}

#[derive(Clone, Copy)]
struct ChiangSetup {
    sigma: DVec3,
    variance: f64,
    azimuthal_scale: f64,
    alpha: f64,
    ior: f64,
    primary_variance: f64,
}

fn chiang_setup(hair: crate::shader::PrincipledHairParameters, sigma: DVec3) -> ChiangSetup {
    let random_factor = random_roughness_factor(hair.random, hair.random_roughness);
    let mut variance = (hair.roughness * random_factor).clamp(0.001, 1.0);
    let mut azimuthal_scale = (hair.radial_roughness * random_factor).clamp(0.001, 1.0);
    let mut primary_variance = ((1.0 - hair.coat.clamp(0.0, 1.0)) * variance).clamp(0.001, 1.0);
    variance = roughness_to_chiang_variance(variance);
    azimuthal_scale = roughness_to_chiang_scale(azimuthal_scale);
    primary_variance = roughness_to_chiang_variance(primary_variance);
    ChiangSetup {
        sigma,
        variance,
        azimuthal_scale,
        alpha: -hair.offset,
        ior: hair.ior,
        primary_variance,
    }
}

fn roughness_to_chiang_variance(roughness: f64) -> f64 {
    let r2 = roughness * roughness;
    let r20 = roughness.powi(20);
    (0.726 * roughness + 0.812 * r2 + 3.700 * r20).powi(2)
}

fn roughness_to_chiang_scale(roughness: f64) -> f64 {
    let r2 = roughness * roughness;
    let r22 = roughness.powi(22);
    (0.265 * roughness + 1.194 * r2 + 5.372 * r22) * (PI / 8.0).sqrt()
}

fn chiang_frame(
    incoming: DVec3,
    tangent: DVec3,
    radial_normal: DVec3,
    hit: HairHitData,
) -> (DVec3, DVec3, DVec3, f64) {
    let x = tangent.normalize_or_zero();
    let y = x.cross(incoming).normalize_or_zero();
    let z = x.cross(y).normalize_or_zero();
    let h = hit
        .ribbon_v
        .map_or_else(|| radial_normal.normalize_or_zero().cross(x).dot(z), |v| -v);
    (x, y, z, h)
}

fn chiang_eval(
    setup: ChiangSetup,
    incoming: DVec3,
    outgoing: DVec3,
    tangent: DVec3,
    radial_normal: DVec3,
    hit: HairHitData,
) -> (DVec3, f64) {
    let (x, y, z, h) = chiang_frame(incoming, tangent, radial_normal, hit);
    let local_o = to_local(incoming, x, y, z);
    let local_i = to_local(outgoing, x, y, z);
    let sin_theta_o = local_o.x;
    let cos_theta_o = cos_from_sin(sin_theta_o);
    let phi_o = local_o.z.atan2(local_o.y);
    let sin_theta_i = local_i.x;
    let cos_theta_i = cos_from_sin(sin_theta_i);
    let phi_i = local_i.z.atan2(local_i.y);
    let cos_theta_t = cos_from_sin(sin_theta_o / setup.ior);
    let sin_gamma_o = h;
    let cos_gamma_o = cos_from_sin(sin_gamma_o);
    let gamma_o = safe_asin(sin_gamma_o);
    let sin_gamma_t =
        sin_gamma_o * cos_theta_o / (setup.ior * setup.ior - sin_theta_o * sin_theta_o).sqrt();
    let cos_gamma_t = cos_from_sin(sin_gamma_t);
    let gamma_t = safe_asin(sin_gamma_t);
    let transmittance = (-setup.sigma * (2.0 * cos_gamma_t / cos_theta_t)).exp();
    let (attenuation, energy) = hair_attenuation(
        fresnel_dielectric_cos(cos_theta_o * cos_gamma_o, setup.ior),
        transmittance,
    );
    let phi = phi_i - phi_o;
    let angles = hair_alpha_angles(sin_theta_o, cos_theta_o, setup.alpha);
    let mut value = DVec3::ZERO;
    let mut pdf = 0.0;
    for p in 0..3 {
        let variance = match p {
            0 => setup.primary_variance,
            1 => 0.25 * setup.variance,
            _ => 4.0 * setup.variance,
        };
        let mp = longitudinal_scattering(
            sin_theta_i,
            cos_theta_i,
            angles[p * 2],
            angles[p * 2 + 1],
            variance,
        );
        let np = azimuthal_scattering(phi, p as i32, setup.azimuthal_scale, gamma_o, gamma_t);
        value += attenuation[p] * (mp * np);
        pdf += energy[p] * mp * np;
    }
    let mp = longitudinal_scattering(
        sin_theta_i,
        cos_theta_i,
        sin_theta_o,
        cos_theta_o,
        4.0 * setup.variance,
    );
    value += attenuation[3] * (mp * M_1_2PI);
    pdf += energy[3] * mp * M_1_2PI;

    (value, pdf)
}

fn chiang_sample(
    setup: ChiangSetup,
    incoming: DVec3,
    tangent: DVec3,
    radial_normal: DVec3,
    hit: HairHitData,
    random: [f64; 4],
) -> HairSample {
    let (x, y, z, h) = chiang_frame(incoming, tangent, radial_normal, hit);
    let local_o = to_local(incoming, x, y, z);
    let sin_theta_o = local_o.x;
    let cos_theta_o = cos_from_sin(sin_theta_o);
    let phi_o = local_o.z.atan2(local_o.y);
    let sin_theta_t = sin_theta_o / setup.ior;
    let cos_theta_t = cos_from_sin(sin_theta_t);
    let sin_gamma_o = h;
    let cos_gamma_o = cos_from_sin(sin_gamma_o);
    let gamma_o = safe_asin(sin_gamma_o);
    let sin_gamma_t =
        sin_gamma_o * cos_theta_o / (setup.ior * setup.ior - sin_theta_o * sin_theta_o).sqrt();
    let cos_gamma_t = cos_from_sin(sin_gamma_t);
    let gamma_t = safe_asin(sin_gamma_t);
    let transmittance = (-setup.sigma * (2.0 * cos_gamma_t / cos_theta_t)).exp();
    let (_, energy) = hair_attenuation(
        fresnel_dielectric_cos(cos_theta_o * cos_gamma_o, setup.ior),
        transmittance,
    );
    let mut p = 0;
    let mut branch = random[2];
    while p < 3 && branch >= energy[p] {
        branch -= energy[p];
        p += 1;
    }
    branch /= energy[p];
    let mut variance = setup.variance;
    if p == 1 {
        variance *= 0.25;
    } else if p >= 2 {
        variance *= 4.0;
    }
    let angles = hair_alpha_angles(sin_theta_o, cos_theta_o, setup.alpha);
    let (sin_theta_o_tilted, cos_theta_o_tilted) = if p < 3 {
        (angles[p * 2], angles[p * 2 + 1])
    } else {
        (sin_theta_o, cos_theta_o)
    };
    branch = branch.max(1.0e-5);
    let fac = 1.0 + variance * (branch + (1.0 - branch) * (-2.0 / variance).exp()).ln();
    let sin_theta_i = -fac * sin_theta_o_tilted
        + (1.0 - fac * fac).max(0.0).sqrt() * (2.0 * PI * random[1]).cos() * cos_theta_o_tilted;
    let phi = if p < 3 {
        delta_phi(p as i32, gamma_o, gamma_t)
            + sample_trimmed_logistic(random[0], setup.azimuthal_scale)
    } else {
        2.0 * PI * random[0]
    };
    let phi_i = phi_o + phi;
    let cos_theta_i = cos_from_sin(sin_theta_i);
    let local_i = DVec3::new(
        sin_theta_i,
        cos_theta_i * phi_i.cos(),
        cos_theta_i * phi_i.sin(),
    );

    let direction = to_global(local_i, x, y, z);
    let (eval, pdf) = chiang_eval(setup, incoming, direction, tangent, radial_normal, hit);
    HairSample::Continuous {
        direction,
        ray: if p == 0 {
            RayKind::Glossy
        } else {
            RayKind::Transmission
        },
        eval,
        pdf,
    }
}

fn hair_attenuation(fresnel: f64, transmittance: DVec3) -> ([DVec3; 4], [f64; 4]) {
    let mut attenuation = [DVec3::ZERO; 4];
    let mut energy = [0.0; 4];
    attenuation[0] = DVec3::splat(fresnel);
    energy[0] = fresnel;
    let mut color = transmittance * ((1.0 - fresnel) * (1.0 - fresnel));
    attenuation[1] = color;
    energy[1] = spectrum_gray(color);
    color *= transmittance * fresnel;
    attenuation[2] = color;
    energy[2] = spectrum_gray(color);
    let numerator = color * transmittance * fresnel;
    let denominator = DVec3::ONE - transmittance * fresnel;
    color = spectrum_safe_divide(numerator, denominator);
    attenuation[3] = color;
    energy[3] = spectrum_gray(color);
    let total = energy.iter().sum::<f64>();
    let factor = safe_divide(1.0, total);
    for sample_energy in &mut energy {
        *sample_energy *= factor;
    }
    (attenuation, energy)
}

fn spectrum_gray(value: DVec3) -> f64 {
    value.dot(REC709_LUMA)
}

fn spectrum_average(value: DVec3) -> f64 {
    (value.x + value.y + value.z) / 3.0
}

fn spectrum_safe_divide(numerator: DVec3, denominator: DVec3) -> DVec3 {
    DVec3::new(
        safe_divide(numerator.x, denominator.x),
        safe_divide(numerator.y, denominator.y),
        safe_divide(numerator.z, denominator.z),
    )
}

fn safe_divide(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn hair_alpha_angles(sin_theta_o: f64, cos_theta_o: f64, alpha: f64) -> [f64; 6] {
    let (sin_1alpha, cos_1alpha) = alpha.sin_cos();
    let sin_2alpha = 2.0 * sin_1alpha * cos_1alpha;
    let cos_2alpha = cos_1alpha * cos_1alpha - sin_1alpha * sin_1alpha;
    let sin_4alpha = 2.0 * sin_2alpha * cos_2alpha;
    let cos_4alpha = cos_2alpha * cos_2alpha - sin_2alpha * sin_2alpha;
    [
        sin_theta_o * cos_2alpha - cos_theta_o * sin_2alpha,
        (cos_theta_o * cos_2alpha + sin_theta_o * sin_2alpha).abs(),
        sin_theta_o * cos_1alpha + cos_theta_o * sin_1alpha,
        (cos_theta_o * cos_1alpha - sin_theta_o * sin_1alpha).abs(),
        sin_theta_o * cos_4alpha + cos_theta_o * sin_4alpha,
        (cos_theta_o * cos_4alpha - sin_theta_o * sin_4alpha).abs(),
    ]
}

fn delta_phi(p: i32, gamma_o: f64, gamma_t: f64) -> f64 {
    2.0 * f64::from(p) * gamma_t - 2.0 * gamma_o + f64::from(p) * PI
}

fn wrap_angle(angle: f64) -> f64 {
    (angle + PI) - 2.0 * PI * ((angle + PI) / (2.0 * PI)).floor() - PI
}

fn logistic(value: f64, scale: f64) -> f64 {
    let exp = (-value.abs() / scale).exp();
    exp / (scale * (1.0 + exp).powi(2))
}

fn logistic_cdf(value: f64, scale: f64) -> f64 {
    let arg = -value / scale;
    if arg > 88.0 {
        0.0
    } else {
        1.0 / (1.0 + arg.exp())
    }
}

fn trimmed_logistic(value: f64, scale: f64) -> f64 {
    let normalization = 1.0 - 2.0 * logistic_cdf(-PI, scale);
    safe_divide(logistic(value, scale), normalization)
}

fn sample_trimmed_logistic(u: f64, scale: f64) -> f64 {
    let cdf_minus_pi = logistic_cdf(-PI, scale);
    let x = -scale * (1.0 / (u * (1.0 - 2.0 * cdf_minus_pi) + cdf_minus_pi) - 1.0).ln();
    x.clamp(-PI, PI)
}

fn azimuthal_scattering(phi: f64, p: i32, scale: f64, gamma_o: f64, gamma_t: f64) -> f64 {
    trimmed_logistic(wrap_angle(phi - delta_phi(p, gamma_o, gamma_t)), scale)
}

fn longitudinal_scattering(
    sin_theta_i: f64,
    cos_theta_i: f64,
    sin_theta_o: f64,
    cos_theta_o: f64,
    variance: f64,
) -> f64 {
    let inv_v = 1.0 / variance;
    let cos_arg = cos_theta_i * cos_theta_o * inv_v;
    let sin_arg = sin_theta_i * sin_theta_o * inv_v;
    if variance <= 0.1 {
        let i0 = log_bessel_i0(cos_arg);
        (i0 - sin_arg - inv_v + 6931.0 / 10_000.0 + (0.5 * inv_v).ln()).exp()
    } else {
        (-sin_arg).exp() * bessel_i0(cos_arg) / (inv_v.sinh() * 2.0 * variance)
    }
}

fn bessel_i0(mut x: f64) -> f64 {
    x *= x;
    let mut value = 1.0 + 0.25 * x;
    let mut power_x_2i = x * x;
    let mut i_fac_2 = 1_u64;
    let mut pow_4_i = 16.0;
    for i in 2..10 {
        i_fac_2 *= (i * i) as u64;
        let new_value = value + power_x_2i / (pow_4_i * i_fac_2 as f64);
        if exact_f64_eq(value, new_value) {
            return value;
        }
        value = new_value;
        power_x_2i *= x;
        pow_4_i *= 4.0;
    }
    value
}

fn log_bessel_i0(x: f64) -> f64 {
    if x > 12.0 {
        x + 0.5 * (1.0 / (8.0 * x) - (2.0 * PI).ln() - x.ln())
    } else {
        bessel_i0(x).ln()
    }
}

fn cos_from_sin(sin: f64) -> f64 {
    (1.0 - sin * sin).max(0.0).sqrt()
}

fn safe_asin(value: f64) -> f64 {
    value.clamp(-1.0, 1.0).asin()
}

fn fresnel_dielectric_cos(cosine: f64, eta: f64) -> f64 {
    let c = cosine.abs();
    let mut g = eta * eta - 1.0 + c * c;
    if g > 0.0 {
        g = g.sqrt();
        let a = (g - c) / (g + c);
        let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
        0.5 * a * a * (1.0 + b * b)
    } else {
        1.0
    }
}

fn classic_hair_eval(
    hair: crate::shader::HairParameters,
    incoming: DVec3,
    outgoing: DVec3,
    radial_normal: DVec3,
) -> (DVec3, f64) {
    let normal = radial_normal.normalize_or_zero();
    let reflection = hair.component == HairComponent::Reflection;
    if (reflection && normal.dot(outgoing) < 0.0) || (!reflection && normal.dot(outgoing) >= 0.0) {
        return (DVec3::ZERO, 0.0);
    }
    let tangent = hair.tangent.normalize_or_zero();
    let roughness1 = hair.roughness_u.clamp(0.001, 1.0);
    let roughness2 = hair.roughness_v.clamp(0.001, 1.0);
    let offset = -hair.offset;
    let iz = tangent.dot(incoming);
    let locy = (incoming - tangent * iz).normalize_or_zero();
    let theta_r = FRAC_PI_2 - iz.clamp(-1.0, 1.0).acos();
    let wo_z = tangent.dot(outgoing);
    let wo_y = (outgoing - tangent * wo_z).normalize_or_zero();
    let theta_i = FRAC_PI_2 - wo_z.clamp(-1.0, 1.0).acos();
    let cosphi_i = wo_y.dot(locy);
    if FRAC_PI_2 - theta_i.abs() < 0.001 || (reflection && cosphi_i < 0.0) {
        return (DVec3::ZERO, 0.0);
    }
    let costheta_i = theta_i.cos();
    let theta_h = f64::midpoint(theta_i, theta_r);
    let t = theta_h - offset;
    let inv_roughness1 = 1.0 / roughness1;
    let (a, b, phi_pdf) = if reflection {
        let a = ((f64::midpoint(FRAC_PI_2, theta_r) - offset) * inv_roughness1).atan();
        let b = ((f64::midpoint(-FRAC_PI_2, theta_r) - offset) * inv_roughness1).atan();
        let mut phi_i = cosphi_i.clamp(-1.0, 1.0).acos() / roughness2;
        phi_i = if phi_i.abs() < PI { phi_i } else { PI };
        (a, b, (0.5 * phi_i).cos() * 0.25 / roughness2)
    } else {
        let a = ((f64::midpoint(FRAC_PI_2, theta_r) - offset) * inv_roughness1).atan();
        let b = ((f64::midpoint(-FRAC_PI_2, theta_r) - offset) * inv_roughness1).atan();
        let c = 2.0 * (FRAC_PI_2 / roughness2).atan();
        let phi = cosphi_i.clamp(-1.0, 1.0).acos().abs();
        let p = PI - phi;
        (a, b, roughness2 / (c * (p * p + roughness2 * roughness2)))
    };
    let theta_pdf = roughness1 / (2.0 * (t * t + roughness1 * roughness1) * (a - b) * costheta_i);
    let pdf = phi_pdf * theta_pdf;
    (DVec3::splat(pdf), pdf)
}

fn classic_hair_sample(
    hair: crate::shader::HairParameters,
    incoming: DVec3,
    color: DVec3,
    random: [f64; 4],
) -> HairSample {
    let tangent = hair.tangent.normalize_or_zero();
    let roughness1 = hair.roughness_u.clamp(0.001, 1.0);
    let roughness2 = hair.roughness_v.clamp(0.001, 1.0);
    let offset = -hair.offset;
    let iz = tangent.dot(incoming);
    let locy = (incoming - tangent * iz).normalize_or_zero();
    let locx = locy.cross(tangent);
    let theta_r = FRAC_PI_2 - iz.clamp(-1.0, 1.0).acos();
    let inv_roughness1 = 1.0 / roughness1;
    let a = ((f64::midpoint(FRAC_PI_2, theta_r) - offset) * inv_roughness1).atan();
    let b = ((f64::midpoint(-FRAC_PI_2, theta_r) - offset) * inv_roughness1).atan();
    let t = roughness1 * (random[0] * (a - b) + b).tan();
    let theta_i = 2.0 * (t + offset) - theta_r;
    let (sin_theta_i, cos_theta_i) = theta_i.sin_cos();
    let (phi, phi_pdf) = if hair.component == HairComponent::Reflection {
        let phi = 2.0 * safe_asin(1.0 - 2.0 * random[1]) * roughness2;
        let phi_pdf = (0.5 * phi).cos() * 0.25 / roughness2;
        (phi, phi_pdf)
    } else {
        let c = 2.0 * (FRAC_PI_2 / roughness2).atan();
        let p = roughness2 * (c * (random[1] - 0.5)).tan();
        let phi_pdf = roughness2 / (c * (p * p + roughness2 * roughness2));
        (p + PI, phi_pdf)
    };
    let (sin_phi, cos_phi) = phi.sin_cos();
    let direction =
        cos_phi * cos_theta_i * locy - sin_phi * cos_theta_i * locx + sin_theta_i * tangent;
    // Keep the source sample PDF: unlike `classic_hair_eval`, it has no cosphi/normal-support rejection.
    let theta_pdf = roughness1 / (2.0 * (t * t + roughness1 * roughness1) * (a - b) * cos_theta_i);
    let pdf = if FRAC_PI_2 - theta_i.abs() < 0.001 {
        0.0
    } else {
        (phi_pdf * theta_pdf).abs()
    };
    HairSample::Continuous {
        direction,
        ray: if hair.component == HairComponent::Reflection {
            RayKind::Glossy
        } else {
            RayKind::Transmission
        },
        // Classic Hair's sample eval is its PDF tinted by Color once; closure weight/alpha are applied by the path.
        eval: DVec3::splat(pdf) * color,
        pdf,
    }
}

fn to_local(vector: DVec3, x: DVec3, y: DVec3, z: DVec3) -> DVec3 {
    DVec3::new(vector.dot(x), vector.dot(y), vector.dot(z))
}

fn to_global(vector: DVec3, x: DVec3, y: DVec3, z: DVec3) -> DVec3 {
    x * vector.x + y * vector.y + z * vector.z
}

#[derive(Clone, Copy)]
struct HuangSetup {
    sigma: DVec3,
    roughness: f64,
    tilt: f64,
    ior: f64,
    aspect_ratio: f64,
    reflection: f64,
    transmission: f64,
    secondary_reflection: f64,
    h: f64,
    normal: DVec3,
    tangent: DVec3,
    bitangent: DVec3,
    wi: DVec3,
    radius: f64,
    e2: f64,
    pixel_coverage: f64,
}

fn huang_setup(
    hair: crate::shader::PrincipledHairParameters,
    sigma: DVec3,
    incoming: DVec3,
    tangent: DVec3,
    radial_normal: DVec3,
    hit: HairHitData,
) -> HuangSetup {
    let y = tangent.normalize_or_zero();
    let x = tangent.cross(incoming).normalize_or_zero();
    let h = hit
        .ribbon_v
        .map_or_else(|| -x.dot(radial_normal.normalize_or_zero()), |v| -v);
    let mut aspect_ratio = hair.aspect_ratio;
    let mut normal = if exact_f64_one(aspect_ratio) {
        x
    } else {
        tangent
            .cross(hit.curve_normal.cross(tangent).normalize_or_zero())
            .normalize_or_zero()
    };
    if aspect_ratio > 1.0 {
        aspect_ratio = 1.0 / aspect_ratio;
        let minor_axis = tangent.cross(normal).normalize_or_zero();
        normal = minor_axis.cross(tangent).normalize_or_zero();
    }
    let (normal, bitangent) = if normal == DVec3::ZERO || !normal.is_finite() {
        let (z, n) = make_orthonormals(y);
        (n, z)
    } else {
        (normal, normal.cross(tangent).normalize_or_zero())
    };
    let wi = to_local(incoming, normal, y, bitangent);
    let e2 = 1.0 - aspect_ratio * aspect_ratio;
    let radius = if e2 == 0.0 {
        1.0
    } else {
        (1.0 - e2 * wi.x * wi.x / (wi.x * wi.x + wi.z * wi.z))
            .max(0.0)
            .sqrt()
    };
    let random_factor = random_roughness_factor(hair.random, hair.random_roughness);
    HuangSetup {
        sigma,
        roughness: (hair.roughness * random_factor).clamp(0.001, 1.0),
        tilt: -hair.offset,
        ior: hair.ior,
        aspect_ratio,
        reflection: hair.reflection.max(0.0),
        transmission: hair.transmission.max(0.0),
        secondary_reflection: hair.secondary_reflection.max(0.0),
        h,
        normal,
        tangent: y,
        bitangent,
        wi,
        radius,
        e2,
        pixel_coverage: hit.pixel_coverage,
    }
}

fn huang_eval(setup: HuangSetup, outgoing: DVec3, rng: &mut u32) -> DVec3 {
    let local_o = to_local(outgoing, setup.normal, setup.tangent, setup.bitangent);
    if setup.tilt.tan() * tan_theta(local_o) < -1.0 {
        return DVec3::ZERO;
    }
    let half_span = (-setup.tilt.tan() * tan_theta(setup.wi)).max(0.0).acos();
    if half_span.is_nan() {
        return DVec3::ZERO;
    }
    let b = setup.aspect_ratio;
    let phi_i = if exact_f64_one(b) {
        0.0
    } else {
        dir_phi(setup.wi)
    };
    let mut h_min = phi_to_h(phi_i + half_span, b, setup.wi);
    let mut h_max = phi_to_h(phi_i - half_span, b, setup.wi);
    let mut dh = 2.0;
    if setup.radius > setup.pixel_coverage {
        let near_min = (-setup.radius).max(setup.h - setup.pixel_coverage);
        let near_max = setup.radius.min(setup.h + setup.pixel_coverage);
        dh = (near_max - near_min).max(0.0) / setup.radius;
        h_min = h_min.max(near_min);
        h_max = h_max.min(near_max);
    }
    h_min = (h_min / setup.radius).max(-0.999);
    h_max = (h_max / setup.radius).min(0.999);
    if h_min >= h_max {
        return DVec3::ZERO;
    }
    let eval_r = huang_eval_r(setup, setup.wi, local_o, h_min, h_max);
    let eval_residual = huang_eval_residual(setup, setup.wi, local_o, h_min, h_max, rng);
    let projected_area = cos_theta(setup.wi) * dh;
    (eval_r + eval_residual) / projected_area
}

fn huang_eval_r(setup: HuangSetup, wi: DVec3, wo: DVec3, h_min: f64, h_max: f64) -> DVec3 {
    if setup.reflection <= 0.0 {
        return DVec3::ZERO;
    }
    let b = setup.aspect_ratio;
    let wh = (wi + wo).normalize_or_zero();
    let alpha = setup.roughness;
    let alpha2 = alpha * alpha;
    let sincos_phi_i = sincos_phi(wi);
    let h_range = h_max - h_min;
    let intervals = 2 * (h_range / (alpha * 0.7) * 0.5).ceil() as usize;
    if intervals == 0 {
        return DVec3::ZERO;
    }
    let res = h_range / intervals as f64;
    let mut integral = 0.0;
    for i in 0..=intervals {
        let h = h_min + i as f64 * res;
        let gamma_m = h_to_gamma(h, b, wi);
        let wm = sphg_dir(setup.tilt, gamma_m, b);
        if microfacet_visible_pair(wi, wo, DVec3::new(wm.x, 0.0, wm.z), wh) {
            let jacobian = d_gamma_d_h(sincos_phi_i, gamma_m, b);
            let weight = if i == 0 || i == intervals {
                0.5
            } else if i % 2 == 0 {
                1.0
            } else {
                2.0
            };
            let cos_mi = wm.dot(wi);
            integral += weight
                * jacobian
                * ggx_d(alpha2, wm.dot(wh))
                * ggx_g(alpha2, cos_mi, wm.dot(wo))
                * arc_length(setup.e2, gamma_m)
                * hair_energy_scale(alpha.sqrt(), cos_mi, setup.ior);
        }
    }
    integral *= (2.0 / 3.0) * res;
    let fresnel = fresnel_dielectric_cos(wi.dot(wh), setup.ior);
    DVec3::splat(setup.reflection * 0.25 * fresnel * integral)
}

fn huang_eval_residual(
    setup: HuangSetup,
    wi: DVec3,
    wo: DVec3,
    h_min: f64,
    h_max: f64,
    rng: &mut u32,
) -> DVec3 {
    if setup.transmission <= 0.0 && setup.secondary_reflection <= 0.0 {
        return DVec3::ZERO;
    }
    let b = setup.aspect_ratio;
    let mu_a = setup.sigma;
    let eta = setup.ior;
    let inv_eta = 1.0 / eta;
    let alpha = setup.roughness;
    let alpha2 = alpha * alpha;
    let sqrt_alpha = alpha.sqrt();
    let sincos_phi_i = sincos_phi(wi);
    let h_range = h_max - h_min;
    let intervals = 2 * (h_range / (alpha * 0.8) * 0.5).ceil() as usize;
    if intervals == 0 {
        return DVec3::ZERO;
    }
    let res = h_range / intervals as f64;
    let mut s_tt = DVec3::ZERO;
    let mut s_trt = DVec3::ZERO;
    let mut s_trrt = DVec3::ZERO;
    for i in 0..=intervals {
        let h = h_min + i as f64 * res;
        let gamma_mi = h_to_gamma(h, b, wi);
        let wmi = sphg_dir(setup.tilt, gamma_mi, b);
        let wmi_ = sphg_dir(0.0, gamma_mi, b);
        let sample1 = [lcg_step_float(rng), lcg_step_float(rng)];
        let wh1 = sample_wh(alpha, wi, wmi, sample1);
        let cos_hi1 = wi.dot(wh1);
        if cos_hi1 <= 0.0 {
            continue;
        }
        let cos_mi1 = wi.dot(wmi);
        let (r1, cos_theta_t1) = fresnel_dielectric(cos_hi1, eta);
        let t1 = 1.0 - r1;
        let scale1 = hair_energy_scale(sqrt_alpha, cos_mi1, eta);
        let wt = refract_angle(wi, wh1, cos_theta_t1, inv_eta);
        let phi_t = dir_phi(wt);
        let gamma_mt = 2.0 * to_phi(phi_t, b) - gamma_mi;
        let wmt = sphg_dir(-setup.tilt, gamma_mt, b);
        let wmt_ = sphg_dir(0.0, gamma_mt, b);
        let cos_mo1 = (-wt).dot(wmi);
        let cos_mi2 = (-wt).dot(wmt);
        let g1o = go(alpha2, cos_mi1, cos_mo1);
        if !microfacet_visible_pair(wi, -wt, wmi, wh1)
            || !microfacet_visible_pair(wi, -wt, wmi_, wh1)
        {
            continue;
        }
        let jacobian = d_gamma_d_h(sincos_phi_i, gamma_mi, b);
        let weight = (if i == 0 || i == intervals {
            0.5
        } else if i % 2 == 0 {
            1.0
        } else {
            2.0
        }) * jacobian;
        let path_t = if exact_f64_one(b) {
            2.0 * (gamma_mi - phi_t).cos()
        } else {
            -point_distance(point(gamma_mi, b), point(gamma_mt + PI, b))
        };
        let a_t = (mu_a * (path_t / cos_theta(wt))).exp();
        let scale2 = hair_energy_scale(sqrt_alpha, cos_mi2, inv_eta);
        if setup.transmission > 0.0 && wo.dot(wt) >= inv_eta - 1e-5 {
            let mut wh2 = -wt + inv_eta * wo;
            let inv_length = 1.0 / wh2.length();
            wh2 *= inv_length;
            let cos_mh2 = wmt.dot(wh2);
            if cos_mh2 >= 0.0 {
                let cos_hi2 = (-wt).dot(wh2);
                let cos_ho2 = (-wo).dot(wh2);
                let cos_mo2 = (-wo).dot(wmt);
                let t2 = (1.0 - fresnel_dielectric_cos(cos_hi2, inv_eta)) * scale2;
                let d2 = ggx_d(alpha2, cos_mh2);
                let g2 = ggx_g(alpha2, cos_mi2, cos_mo2);
                let result = a_t
                    * (weight * t1 * scale1 * t2 * d2 * g1o * g2 / cos_mo1
                        * cos_mi1
                        * cos_hi2
                        * cos_ho2
                        * inv_length
                        * inv_length);
                if result.is_finite() {
                    s_tt += result * (setup.transmission * arc_length(setup.e2, gamma_mt));
                }
            }
        }
        if setup.secondary_reflection <= 0.0 {
            continue;
        }
        let sample2 = [lcg_step_float(rng), lcg_step_float(rng)];
        let wh2 = sample_wh(alpha, -wt, wmt, sample2);
        let cos_hi2 = (-wt).dot(wh2);
        if cos_hi2 <= 0.0 {
            continue;
        }
        let r2 = fresnel_dielectric_cos(cos_hi2, inv_eta);
        let wtr = -reflect(wt, wh2);
        if (-wtr).dot(wo) < inv_eta - 1e-5 {
            s_trrt += DVec3::splat(weight) * huang_eval_trrt(t1, r2, a_t);
            continue;
        }
        if !microfacet_visible_pair(-wt, -wtr, wmt, wh2)
            || !microfacet_visible_pair(-wt, -wtr, wmt_, wh2)
        {
            continue;
        }
        let phi_tr = dir_phi(wtr);
        let gamma_mtr = gamma_mi - 2.0 * (to_phi(phi_t, b) - to_phi(phi_tr, b)) + PI;
        let wmtr = sphg_dir(-setup.tilt, gamma_mtr, b);
        let wmtr_ = sphg_dir(0.0, gamma_mtr, b);
        let mut wh3 = wtr + inv_eta * wo;
        let inv_length3 = 1.0 / wh3.length();
        wh3 *= inv_length3;
        let cos_mh3 = wmtr.dot(wh3);
        if cos_mh3 < 0.0
            || !microfacet_visible_pair(wtr, -wo, wmtr, wh3)
            || !microfacet_visible_pair(wtr, -wo, wmtr_, wh3)
        {
            s_trrt += DVec3::splat(weight) * huang_eval_trrt(t1, r2, a_t);
            continue;
        }
        let cos_hi3 = wh3.dot(wtr);
        let cos_ho3 = wh3.dot(-wo);
        let cos_mi3 = wmtr.dot(wtr);
        let t3 = (1.0 - fresnel_dielectric_cos(cos_hi3, inv_eta))
            * hair_energy_scale(sqrt_alpha, cos_mi3, inv_eta);
        let d3 = ggx_d(alpha2, cos_mh3);
        let path_tr = if exact_f64_one(b) {
            -2.0 * (phi_tr - gamma_mt).cos().abs()
        } else {
            -point_distance(point(gamma_mtr, b), point(gamma_mt, b))
        };
        let a_tr = (mu_a * (path_tr / cos_theta(wtr))).exp();
        let cos_mo2 = wmt.dot(-wtr);
        let g2o = go(alpha2, cos_mi2, cos_mo2);
        let g3 = ggx_g(alpha2, cos_mi3, wmtr.dot(-wo));
        let result = (a_t * a_tr)
            * (weight * t1 * scale1 * r2 * scale2 * t3 * d3 * g1o * g2o * g3 / (cos_mo1 * cos_mo2)
                * cos_mi1
                * cos_mi2
                * cos_hi3
                * cos_ho3
                * inv_length3
                * inv_length3);
        if result.is_finite() {
            s_trt += result * (setup.secondary_reflection * arc_length(setup.e2, gamma_mtr));
        }
        s_trrt += DVec3::splat(weight) * huang_eval_trrt(t1, r2, a_t);
    }
    let m = longitudinal_scattering(
        sin_theta(wi),
        cos_theta(wi),
        sin_theta(wo),
        cos_theta(wo),
        4.0 * setup.roughness,
    );
    let simpson = (2.0 / 3.0) * res;
    ((s_tt + s_trt) * (inv_eta * inv_eta) + s_trrt * (m * M_1_2PI * M_2_PI)) * simpson
}

fn huang_eval_trrt(t: f64, r: f64, attenuation: DVec3) -> DVec3 {
    let t_avg = (1.0 - r).max(1.0e-5);
    let trrt = attenuation * attenuation * attenuation * (t * r * r * t_avg);
    spectrum_safe_divide(trrt, DVec3::ONE - attenuation * (1.0 - t_avg))
}

fn huang_sample(setup: HuangSetup, random: [f64; 4], rng: &mut u32) -> HairSample {
    if setup.h.abs() >= setup.radius {
        return HairSample::Absorbed;
    }
    let roughness = setup.roughness;
    let sample_lobe = random[0];
    let sample_h = random[1];
    let sample_h1 = [random[2], lcg_step_float(rng)];
    let sample_h2 = [lcg_step_float(rng), lcg_step_float(rng)];
    let sample_h3 = [lcg_step_float(rng), lcg_step_float(rng)];
    let wi = setup.wi;
    let b = setup.aspect_ratio;
    let h_div_r = if setup.radius > setup.pixel_coverage {
        setup.h / setup.radius
    } else {
        sample_h * 2.0 - 1.0
    };
    let gamma_mi = h_to_gamma(h_div_r, b, wi);
    let wmi_ = sphg_dir(0.0, gamma_mi, b);
    let (st, ct) = setup.tilt.sin_cos();
    let wmi = DVec3::new(wmi_.x * ct, st, wmi_.z * ct);
    let cos_mi1 = wmi.dot(wi);
    if cos_mi1 < 0.0 || wmi_.dot(wi) < 0.0 {
        return HairSample::Absorbed;
    }
    let roughness2 = roughness * roughness;
    let sqrt_roughness = roughness.sqrt();
    let wh1 = sample_wh(roughness, wi, wmi, sample_h1);
    let wr = -reflect(wi, wh1);
    if !microfacet_visible(wi, wmi_, wh1) {
        return HairSample::Absorbed;
    }
    let (r1, cos_theta_t1) = fresnel_dielectric(wi.dot(wh1), setup.ior);
    let scale1 = hair_energy_scale(sqrt_roughness, cos_mi1, setup.ior);
    let r = setup.reflection
        * r1
        * scale1
        * (if microfacet_visible(wr, wmi_, wh1) {
            1.0
        } else {
            0.0
        })
        * go(roughness2, cos_mi1, wmi.dot(wr));
    let inv_eta = 1.0 / setup.ior;
    let wt = refract_angle(wi, wh1, cos_theta_t1, inv_eta);
    let phi_t = dir_phi(wt);
    let gamma_mt = 2.0 * to_phi(phi_t, b) - gamma_mi;
    let wmt = sphg_dir(-setup.tilt, gamma_mt, b);
    let wmt_ = sphg_dir(0.0, gamma_mt, b);
    let wh2 = sample_wh(roughness, -wt, wmt, sample_h2);
    let wtr = -reflect(wt, wh2);
    let mut tt = DVec3::ZERO;
    let mut trt = DVec3::ZERO;
    let mut trrt = DVec3::ZERO;
    let cos_mi2 = (-wt).dot(wmt);
    let mut wtt = DVec3::ZERO;
    let mut wtrt = DVec3::ZERO;
    let mut wtrrt = DVec3::ZERO;
    if cos_mi2 > 0.0 && microfacet_visible(-wt, wmi_, wh1) && microfacet_visible(-wt, wmt_, wh2) {
        let path_t = if exact_f64_one(b) {
            2.0 * (phi_t - gamma_mi).cos()
        } else {
            -point_distance(point(gamma_mi, b), point(gamma_mt + PI, b))
        };
        let a_t = (setup.sigma * (path_t / cos_theta(wt))).exp();
        let (r2, cos_theta_t2) = fresnel_dielectric((-wt).dot(wh2), inv_eta);
        let t1 = (1.0 - r1) * scale1 * go(roughness2, cos_mi1, wmi.dot(-wt));
        let t2 = 1.0 - r2;
        let scale2 = hair_energy_scale(sqrt_roughness, cos_mi2, inv_eta);
        wtt = refract_angle(-wt, wh2, cos_theta_t2, setup.ior);
        if wmt.dot(-wtt) > 0.0 && t2 > 0.0 && microfacet_visible(-wtt, wmt_, wh2) {
            tt = DVec3::splat(
                setup.transmission * t1 * t2 * scale2 * go(roughness2, cos_mi2, wmt.dot(-wtt)),
            ) * a_t;
        }
        let phi_tr = dir_phi(wtr);
        let gamma_mtr = gamma_mi - 2.0 * (to_phi(phi_t, b) - to_phi(phi_tr, b)) + PI;
        let wmtr = sphg_dir(-setup.tilt, gamma_mtr, b);
        let wh3 = sample_wh(roughness, wtr, wmtr, sample_h3);
        let (r3, cos_theta_t3) = fresnel_dielectric(wtr.dot(wh3), inv_eta);
        wtrt = refract_angle(wtr, wh3, cos_theta_t3, setup.ior);
        let cos_mi3 = wmtr.dot(wtr);
        if cos_mi3 > 0.0 {
            let path_tr = if exact_f64_one(b) {
                -2.0 * (phi_tr - gamma_mt).cos().abs()
            } else {
                -point_distance(point(gamma_mt, b), point(gamma_mtr, b))
            };
            let a_tr = (setup.sigma * (path_tr / cos_theta(wtr))).exp();
            let tr = a_t
                * a_tr
                * (t1
                    * r2
                    * scale2
                    * hair_energy_scale(sqrt_roughness, cos_mi3, inv_eta)
                    * go(roughness2, cos_mi2, wmt.dot(-wtr)));
            let t3 = 1.0 - r3;
            if t3 > 0.0 && microfacet_visible_pair(wtr, -wtrt, DVec3::new(wmtr.x, 0.0, wmtr.z), wh3)
            {
                trt = tr
                    * (setup.secondary_reflection * t3 * go(roughness2, cos_mi3, wmtr.dot(-wtrt)));
            }
            let rand_theta = lcg_step_float(rng).max(1.0e-5);
            let fac = 1.0
                + 4.0
                    * setup.roughness
                    * (rand_theta + (1.0 - rand_theta) * (-0.5 / setup.roughness).exp()).ln();
            let sin_theta_o = -fac * sin_theta(wi)
                + cos_from_sin(fac) * (2.0 * PI * lcg_step_float(rng)).cos() * cos_theta(wi);
            let cos_theta_o = cos_from_sin(sin_theta_o);
            let phi_o = 2.0 * PI * lcg_step_float(rng);
            wtrrt = DVec3::new(
                phi_o.sin() * cos_theta_o,
                sin_theta_o,
                phi_o.cos() * cos_theta_o,
            );
            let a_avg = (a_t * a_tr).sqrt();
            let t_avg = f64::midpoint(1.0 - r2, 1.0 - r3).max(1.0e-5);
            let a_res = spectrum_safe_divide(a_avg * (t_avg), DVec3::ONE - a_avg * (1.0 - t_avg));
            trrt = tr * r3 * go(roughness2, cos_mi3, wmtr.dot(-reflect(wtr, wh3))) * a_res;
        }
    }
    let tt_energy = spectrum_average(tt);
    let trt_energy = spectrum_average(trt);
    let trrt_energy = spectrum_average(trrt);
    let total_energy = r + tt_energy + trt_energy + trrt_energy;
    if total_energy == 0.0 {
        return HairSample::Absorbed;
    }
    let branch = sample_lobe * total_energy;
    let (local_out, eval) = if branch < r {
        (wr, DVec3::splat(total_energy))
    } else if branch < r + tt_energy {
        (wtt, tt / tt_energy * total_energy)
    } else if branch < r + tt_energy + trt_energy {
        (wtrt, trt / trt_energy * total_energy)
    } else {
        (wtrrt, trrt / trrt_energy * total_energy)
    };
    HairSample::Continuous {
        direction: to_global(local_out, setup.normal, setup.tangent, setup.bitangent),
        ray: RayKind::Glossy,
        eval,
        pdf: 1.0,
    }
}

fn sample_wh(roughness: f64, wi: DVec3, wm: DVec3, random: [f64; 2]) -> DVec3 {
    let (s, t) = make_orthonormals(wm);
    let wi_wm = to_local(wi, s, t, wm);
    let wh_wm =
        super::principled_bsdf::sample_visible_normal(wi_wm, roughness, random[0], random[1]);
    to_global(wh_wm, s, t, wm)
}

fn microfacet_visible(v: DVec3, m: DVec3, h: DVec3) -> bool {
    v.dot(h) > 0.0 && v.dot(m) > 0.0
}

fn microfacet_visible_pair(wi: DVec3, wo: DVec3, m: DVec3, h: DVec3) -> bool {
    microfacet_visible(wi, m, h) && microfacet_visible(wo, m, h)
}

fn go(alpha2: f64, cos_ni: f64, cos_no: f64) -> f64 {
    (1.0 + ggx_lambda(alpha2, cos_ni))
        / (1.0 + ggx_lambda(alpha2, cos_ni) + ggx_lambda(alpha2, cos_no))
}

fn ggx_g(alpha2: f64, cos_ni: f64, cos_no: f64) -> f64 {
    1.0 / (1.0 + ggx_lambda(alpha2, cos_ni) + ggx_lambda(alpha2, cos_no))
}

fn ggx_lambda(alpha2: f64, cosine: f64) -> f64 {
    let tan_squared = (1.0 / (cosine * cosine) - 1.0).max(0.0);
    0.5 * ((1.0 + alpha2 * tan_squared).sqrt() - 1.0)
}

fn ggx_d(alpha2: f64, cos_nh: f64) -> f64 {
    let cos_nh2 = (cos_nh * cos_nh).min(1.0);
    let one_minus = 1.0 - cos_nh2;
    alpha2 / (PI * (one_minus + alpha2 * cos_nh2).powi(2))
}

fn hair_energy_scale(rough: f64, mu: f64, ior: f64) -> f64 {
    let z = ((ior - 1.0) / (ior + 1.0)).abs().sqrt();
    let energy = if ior < 1.0 {
        super::principled_tables::ggx_glass_inverse_energy(rough, mu, z)
    } else {
        super::principled_tables::ggx_glass_energy(rough, mu, z)
    };
    1.0 / energy
}

fn fresnel_dielectric(cosine_incident: f64, eta: f64) -> (f64, f64) {
    let cos_i_abs = cosine_incident.abs();
    let eta_cos_t_squared = eta * eta - (1.0 - cos_i_abs * cos_i_abs);
    if eta_cos_t_squared <= 0.0 {
        return (1.0, 0.0);
    }
    let cos_i = cosine_incident.abs();
    let cos_t = -eta_cos_t_squared.sqrt() / eta;
    let r_s = (cos_i + eta * cos_t) / (cos_i - eta * cos_t);
    let r_p = (cos_t + eta * cos_i) / (eta * cos_i - cos_t);
    (f64::midpoint(r_s * r_s, r_p * r_p), cos_t)
}

fn refract_angle(incident: DVec3, normal: DVec3, cos_theta_t: f64, inv_eta: f64) -> DVec3 {
    normal * (inv_eta * normal.dot(incident) + cos_theta_t) - incident * inv_eta
}

fn reflect(incident: DVec3, normal: DVec3) -> DVec3 {
    incident - normal * (2.0 * normal.dot(incident))
}

fn sin_theta(direction: DVec3) -> f64 {
    direction.y
}
fn cos_theta(direction: DVec3) -> f64 {
    (direction.x * direction.x + direction.z * direction.z)
        .max(0.0)
        .sqrt()
}
fn tan_theta(direction: DVec3) -> f64 {
    sin_theta(direction) / cos_theta(direction)
}
fn dir_phi(direction: DVec3) -> f64 {
    direction.x.atan2(direction.z)
}
fn sincos_phi(direction: DVec3) -> (f64, f64) {
    let cosine = cos_theta(direction);
    (direction.x / cosine, direction.z / cosine)
}
fn point(gamma: f64, b: f64) -> (f64, f64) {
    (gamma.sin(), b * gamma.cos())
}
fn point_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}
fn arc_length(e2: f64, gamma: f64) -> f64 {
    if e2 == 0.0 {
        1.0
    } else {
        (1.0 - e2 * gamma.sin().powi(2)).sqrt()
    }
}

fn to_phi(gamma: f64, b: f64) -> f64 {
    if exact_f64_one(b) {
        gamma
    } else {
        (b * gamma.sin()).atan2(gamma.cos())
    }
}

fn to_gamma(phi: f64, b: f64) -> f64 {
    if exact_f64_one(b) {
        phi
    } else {
        phi.sin().atan2(b * phi.cos())
    }
}

fn phi_to_h(phi: f64, b: f64, wi: DVec3) -> f64 {
    if exact_f64_one(b) {
        return -phi.sin();
    }
    let gamma = to_gamma(phi, b);
    let (sin_gamma, cos_gamma) = gamma.sin_cos();
    let (sin_phi_i, cos_phi_i) = sincos_phi(wi);
    -cos_phi_i * sin_gamma + b * sin_phi_i * cos_gamma
}

fn h_to_gamma(h_div_r: f64, b: f64, wi: DVec3) -> f64 {
    if exact_f64_one(b) {
        -h_div_r.asin()
    } else {
        wi.z.atan2(-b * wi.x) - (-h_div_r).acos()
    }
}

fn d_gamma_d_h(sincos_phi_i: (f64, f64), gamma: f64, b: f64) -> f64 {
    if exact_f64_one(b) {
        1.0 / gamma.cos()
    } else {
        let (sin_gamma, cos_gamma) = gamma.sin_cos();
        1.0 / (sincos_phi_i.1 * cos_gamma + b * sincos_phi_i.0 * sin_gamma)
    }
}

fn sphg_dir(theta: f64, gamma: f64, b: f64) -> DVec3 {
    let (sin_theta, cos_theta) = theta.sin_cos();
    let (sin_gamma, cos_gamma) = gamma.sin_cos();
    let (sin_phi, cos_phi) = if exact_f64_one(b) || cos_gamma.abs() < 1.0e-6 {
        (sin_gamma, cos_gamma)
    } else {
        let tan_phi = b * sin_gamma / cos_gamma;
        let cos_phi = cos_gamma.signum() / (tan_phi * tan_phi + 1.0).sqrt();
        (cos_phi * tan_phi, cos_phi)
    };
    DVec3::new(sin_phi * cos_theta, sin_theta, cos_phi * cos_theta)
}

fn lcg_step_float(rng: &mut u32) -> f64 {
    *rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    ((*rng as f32) * (1.0_f32 / u32::MAX as f32)) as f64
}

#[cfg(test)]
mod tests {
    use super::{
        HairComponent, classic_hair_eval, logistic_cdf, sample_trimmed_logistic, wrap_angle,
    };
    use crate::shader::HairParameters;
    use glam::DVec3;
    use std::f64::consts::PI;

    #[test]
    fn trimmed_logistic_sample_stays_inside_its_support() {
        for u in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let sample = sample_trimmed_logistic(u, 0.35);
            assert!((-PI..=PI).contains(&sample));
        }
        assert!((logistic_cdf(-PI, 0.35) - (1.0 - logistic_cdf(PI, 0.35))).abs() < 1.0e-12);
    }

    #[test]
    fn wrapped_hair_azimuth_is_periodic() {
        let angle = 0.7;
        assert!((wrap_angle(angle) - wrap_angle(angle + 4.0 * PI)).abs() < 1.0e-12);
    }

    #[test]
    fn classic_reflection_eval_is_zero_below_radial_hemisphere() {
        let hair = HairParameters {
            component: HairComponent::Reflection,
            offset: 0.0,
            roughness_u: 0.2,
            roughness_v: 0.4,
            tangent: DVec3::Z,
        };
        let (value, pdf) = classic_hair_eval(hair, DVec3::Y, -DVec3::X, DVec3::X);
        assert_eq!(value, DVec3::ZERO);
        assert_eq!(pdf, 0.0);
    }
}
