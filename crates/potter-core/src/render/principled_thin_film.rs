//! Cycles 5.2.2's Belcour–Barla thin-film Fresnel, used by the Principled path-tracer lobes.
//!
//! The sensitivity data is converted to Rec.709 and DC-normalized from `scene/shader.tables`;
//! the Fresnel and three-order Airy sum follow `kernel/closure/bsdf_util.h`.

use std::f32::consts::TAU;

use glam::DVec3;

use super::principled_thin_film_table::THIN_FILM_CMF;

pub(super) const THICKNESS_CUTOFF: f64 = 0.1_f32 as f64;

/// Spectral dielectric-film reflectance, rescaled to the Principled F0 and its transmitted cosine.
pub(super) fn dielectric_fresnel(
    cosine_incident: f64,
    thickness: f64,
    film_ior: f64,
    substrate_ior: f64,
    f0: DVec3,
) -> (DVec3, f64) {
    let cosine_incident = cosine_incident as f32;
    let thickness = thickness as f32;
    let film_ior = film_ior as f32;
    let substrate_ior = substrate_ior as f32;
    let (red, cos_theta_t) = iridescence_channel(
        0,
        cosine_incident,
        thickness,
        film_ior,
        substrate_ior,
        0.0,
        -1.0,
    );
    let green = iridescence_channel(
        1,
        cosine_incident,
        thickness,
        film_ior,
        substrate_ior,
        0.0,
        -1.0,
    )
    .0;
    let blue = iridescence_channel(
        2,
        cosine_incident,
        thickness,
        film_ior,
        substrate_ior,
        0.0,
        -1.0,
    )
    .0;
    let mut channels = [red, green, blue];
    let f0_ratio = (substrate_ior - 1.0) / (substrate_ior + 1.0);
    let f0_real = f0_ratio * f0_ratio;
    if f0_real > 1.0e-5
        && !channels
            .iter()
            .all(|channel| crate::float::equal_f32(*channel, 1.0))
    {
        let f0 = f0.to_array().map(|channel| channel as f32);
        for (channel, f0) in channels.iter_mut().zip(f0) {
            let strength = saturate((*channel - 1.0) / (f0_real - 1.0));
            *channel *= 1.0 + (f0 / f0_real - 1.0) * strength;
        }
    }
    (
        DVec3::from_array(channels.map(f64::from)),
        f64::from(cos_theta_t),
    )
}

/// Spectral film reflectance for Cycles' F82 metallic Fresnel model.
pub(super) fn f82_fresnel(
    cosine_incident: f64,
    thickness: f64,
    film_ior: f64,
    f0: DVec3,
    b: DVec3,
) -> DVec3 {
    let f0 = f0.to_array().map(|channel| channel as f32);
    let b = b.to_array().map(|channel| channel as f32);
    DVec3::from_array([
        f64::from(f82_iridescence_channel(
            0,
            cosine_incident as f32,
            thickness as f32,
            film_ior as f32,
            f0[0],
            b[0],
        )),
        f64::from(f82_iridescence_channel(
            1,
            cosine_incident as f32,
            thickness as f32,
            film_ior as f32,
            f0[1],
            b[1],
        )),
        f64::from(f82_iridescence_channel(
            2,
            cosine_incident as f32,
            thickness as f32,
            film_ior as f32,
            f0[2],
            b[2],
        )),
    ])
}

fn f82_iridescence_channel(
    channel: usize,
    cosine_incident: f32,
    thickness: f32,
    film_ior: f32,
    f0: f32,
    b: f32,
) -> f32 {
    let f82 = f82_fresnel_at(1.0 / 7.0, f0, b);
    let r = f0.min(0.999);
    let sqrt_r = r.sqrt();
    let n = ((1.0 + sqrt_r) / (1.0 - sqrt_r)) * (1.0 - f82) + ((1.0 - r) / (1.0 + r)) * f82;
    let n_plus_one = n + 1.0;
    let n_minus_one = n - 1.0;
    let k = ((r * n_plus_one * n_plus_one - n_minus_one * n_minus_one) / (1.0 - r))
        .max(0.0)
        .sqrt();
    iridescence_channel(channel, cosine_incident, thickness, film_ior, n, k, f82).0
}

#[derive(Clone, Copy)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    fn multiply(self, other: Self) -> Self {
        let im = self.re * other.im + self.im * other.re;
        Self {
            re: self.re * other.re - self.im * other.im,
            im,
        }
    }
    fn scale(self, value: f32) -> Self {
        Self {
            re: self.re * value,
            im: self.im * value,
        }
    }
}

/// Cycles' `fresnel_iridescence_channel` for one CMF channel; `f82 < 0` selects a dielectric.
fn iridescence_channel(
    channel: usize,
    cosine_incident: f32,
    thickness: f32,
    film_ior: f32,
    substrate_n: f32,
    substrate_k: f32,
    f82: f32,
) -> (f32, f32) {
    let film_ior = if thickness < 1.0 {
        let x = saturate(thickness);
        let smooth = x * x * (3.0 - 2.0 * x);
        1.0 + (film_ior - 1.0) * smooth
    } else {
        film_ior
    };
    let Some((r12_s, r12_p, cos_theta_2, phase12_s, phase12_p)) =
        dielectric_interface(cosine_incident, film_ior)
    else {
        return (1.0, 0.0);
    };

    let (r23_s, r23_p, phase23_s, phase23_p, cos_theta_3) = if f82 >= 0.0 {
        let (r23_s, r23_p, phase_s, phase_p) =
            conductor_interface(-cos_theta_2, film_ior, substrate_n, substrate_k, f82);
        (r23_s, r23_p, phase_s, phase_p, 0.0)
    } else {
        let Some((r23_s, r23_p, cos_theta_3, phase_s, phase_p)) =
            dielectric_interface(-cos_theta_2, substrate_n / film_ior)
        else {
            return (1.0, 0.0);
        };
        (
            r23_s,
            r23_p,
            Complex {
                re: phase_s,
                im: 0.0,
            },
            Complex {
                re: phase_p,
                im: 0.0,
            },
            cos_theta_3,
        )
    };

    let opd = -2.0 * film_ior * thickness * cos_theta_2;
    let phase_s = phase23_s.scale(-phase12_s);
    let phase_p = phase23_p.scale(-phase12_p);
    let reflectance_s = airy_summation(channel, r12_s, r23_s, opd, phase_s);
    let reflectance_p = airy_summation(channel, r12_p, r23_p, opd, phase_p);
    (saturate(reflectance_s.midpoint(reflectance_p)), cos_theta_3)
}

/// Squared polarized dielectric reflectances, negative transmitted cosine, and phase signs.
fn dielectric_interface(cosine_incident: f32, eta: f32) -> Option<(f32, f32, f32, f32, f32)> {
    let cosine_incident = cosine_incident.abs();
    let eta_cos_theta_t_squared = eta * eta - (1.0 - cosine_incident * cosine_incident);
    if eta_cos_theta_t_squared <= 0.0 {
        return None;
    }
    let cos_theta_t = -eta_cos_theta_t_squared.sqrt() / eta;
    let r_s = (cosine_incident + eta * cos_theta_t) / (cosine_incident - eta * cos_theta_t);
    let r_p = (cos_theta_t + eta * cosine_incident) / (eta * cosine_incident - cos_theta_t);
    Some((
        r_s * r_s,
        r_p * r_p,
        cos_theta_t,
        if r_s >= 0.0 { 1.0 } else { -1.0 },
        if r_p >= 0.0 { 1.0 } else { -1.0 },
    ))
}

/// Cycles' F82 conductor interface reflectance and physical-conductor reflection phase.
fn conductor_interface(
    cosine_incident: f32,
    ambient_ior: f32,
    substrate_n: f32,
    substrate_k: f32,
    f82: f32,
) -> (f32, f32, Complex, Complex) {
    let eta1 = ambient_ior;
    let eta2 = substrate_n;
    let k2 = substrate_k;
    let eta1_squared = eta1 * eta1;
    let eta2_squared = eta2 * eta2;
    let k2_squared = k2 * k2;
    let two_eta2_k2 = 2.0 * eta2 * k2;
    let t1 = eta2_squared - k2_squared - eta1_squared * (1.0 - cosine_incident * cosine_incident);
    let t2 = (t1 * t1 + two_eta2_k2 * two_eta2_k2).sqrt();
    let u_squared = t2.midpoint(t1).max(0.0);
    let v_squared = t2.midpoint(-t1).max(0.0);
    let u = u_squared.sqrt();
    let v = v_squared.sqrt();

    let n = eta2 / eta1;
    let k_over_eta = k2 / eta1;
    let k_squared = k_over_eta * k_over_eta;
    let n_minus_one = n - 1.0;
    let n_plus_one = n + 1.0;
    let n2_minus_one = n_minus_one * n_minus_one;
    let n2_plus_one = n_plus_one * n_plus_one;
    let f0 = (n2_minus_one + k_squared) / (n2_plus_one + k_squared);
    let b = f82_b(f0, f82);
    let reflectance = f82_fresnel_at(cosine_incident, f0, b);

    let eta1_cosine = eta1 * cosine_incident;
    let re_s = -u_squared - v_squared + eta1_cosine * eta1_cosine;
    let im_s = -2.0 * eta1 * cosine_incident * v;
    let phase_s = unit_complex(re_s, im_s);
    let t3 = (eta2_squared + k2_squared) * cosine_incident;
    let re_p = t3 * t3 - eta1_squared * (u_squared + v_squared);
    let im_p = 2.0 * eta1 * cosine_incident * (two_eta2_k2 * u - (eta2_squared - k2_squared) * v);
    let phase_p = unit_complex(re_p, im_p);
    (reflectance, reflectance, phase_s, phase_p)
}

fn unit_complex(re: f32, im: f32) -> Complex {
    let magnitude = (re * re + im * im).sqrt();
    if magnitude == 0.0 {
        Complex { re: 1.0, im: 0.0 }
    } else {
        Complex {
            re: re / magnitude,
            im: im / magnitude,
        }
    }
}

fn f82_b(f0: f32, f82: f32) -> f32 {
    let f = 6.0 / 7.0;
    let f2 = f * f;
    let f5 = f2 * f2 * f;
    (7.0 / (f5 * f)) * (f0 + (1.0 - f0) * f5 - f82)
}

fn f82_fresnel_at(cosine: f32, f0: f32, b: f32) -> f32 {
    let s = saturate(1.0 - cosine);
    let s2 = s * s;
    let s5 = s2 * s2 * s;
    saturate(f0 + (1.0 - f0) * s5 - b * cosine * s5 * s)
}

fn airy_summation(channel: usize, r12: f32, r23: f32, opd: f32, phasor: Complex) -> f32 {
    let t121 = 1.0 - r12;
    let r123 = r12 * r23;
    let r123_root = r123.sqrt();
    let rs = t121 * t121 * r23 / (1.0 - r123);
    let mut result = rs + r12;
    let mut coefficient = rs - t121;
    let mut accumulator = phasor;
    for order in 1..4 {
        coefficient *= r123_root;
        let sensitivity = sensitivity(channel, order as f32 * opd);
        result +=
            coefficient * 2.0 * (accumulator.re * sensitivity.re + accumulator.im * sensitivity.im);
        accumulator = accumulator.multiply(phasor);
    }
    result
}

fn sensitivity(channel: usize, opd: f32) -> Complex {
    let x = saturate(TAU * opd / 60_000.0) * 511.0;
    let index = (x as usize).min(511);
    let next = (index + 1).min(511);
    let fraction = x - index as f32;
    let row0 = THIN_FILM_CMF[index];
    let row1 = THIN_FILM_CMF[next];
    Complex {
        re: (1.0 - fraction) * f32::from_bits(row0[channel])
            + fraction * f32::from_bits(row1[channel]),
        im: (1.0 - fraction) * f32::from_bits(row0[channel + 3])
            + fraction * f32::from_bits(row1[channel + 3]),
    }
}

fn saturate(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}
