//! Cycles 5.2.2's Belcour–Barla thin-film Fresnel, used by the Principled path-tracer lobes.
//!
//! The sensitivity data is converted to Rec.709 and DC-normalized from `scene/shader.tables`;
//! the Fresnel and three-order Airy sum follow `kernel/closure/bsdf_util.h`.

use std::f64::consts::TAU;

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
    let (red, cos_theta_t) = iridescence_channel(
        0,
        cosine_incident,
        thickness,
        film_ior,
        substrate_ior,
        0.0,
        -1.0,
    );
    let (green, _) = iridescence_channel(
        1,
        cosine_incident,
        thickness,
        film_ior,
        substrate_ior,
        0.0,
        -1.0,
    );
    let (blue, _) = iridescence_channel(
        2,
        cosine_incident,
        thickness,
        film_ior,
        substrate_ior,
        0.0,
        -1.0,
    );
    let mut result = DVec3::new(red, green, blue);
    let f0_real = ((substrate_ior - 1.0) / (substrate_ior + 1.0)).powi(2);
    if f0_real > 1.0e-5 && result != DVec3::ONE {
        let f0 = f0.to_array();
        let mut channels = result.to_array();
        for (channel, f0) in channels.iter_mut().zip(f0) {
            let strength = saturate((*channel - 1.0) / (f0_real - 1.0));
            *channel *= 1.0 + (f0 / f0_real - 1.0) * strength;
        }
        result = DVec3::from_array(channels);
    }
    (result, cos_theta_t)
}

/// Spectral film reflectance for Cycles' F82 metallic Fresnel model.
pub(super) fn f82_fresnel(
    cosine_incident: f64,
    thickness: f64,
    film_ior: f64,
    f0: DVec3,
    b: DVec3,
) -> DVec3 {
    let f0 = f0.to_array();
    let b = b.to_array();
    DVec3::new(
        f82_iridescence_channel(0, cosine_incident, thickness, film_ior, f0[0], b[0]),
        f82_iridescence_channel(1, cosine_incident, thickness, film_ior, f0[1], b[1]),
        f82_iridescence_channel(2, cosine_incident, thickness, film_ior, f0[2], b[2]),
    )
}

fn f82_iridescence_channel(
    channel: usize,
    cosine_incident: f64,
    thickness: f64,
    film_ior: f64,
    f0: f64,
    b: f64,
) -> f64 {
    let f82 = f82_fresnel_at(1.0 / 7.0, f0, b);
    let r = f0.min(0.999);
    let sqrt_r = r.sqrt();
    let n = ((1.0 + sqrt_r) / (1.0 - sqrt_r)) * (1.0 - f82) + ((1.0 - r) / (1.0 + r)) * f82;
    let k = ((r * (n + 1.0).powi(2) - (n - 1.0).powi(2)) / (1.0 - r))
        .max(0.0)
        .sqrt();
    iridescence_channel(channel, cosine_incident, thickness, film_ior, n, k, f82).0
}

#[derive(Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn multiply(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

/// Cycles' `fresnel_iridescence_channel` for one CMF channel; `f82 < 0` selects a dielectric.
fn iridescence_channel(
    channel: usize,
    cosine_incident: f64,
    thickness: f64,
    film_ior: f64,
    substrate_n: f64,
    substrate_k: f64,
    f82: f64,
) -> (f64, f64) {
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
    let phase_s = phase23_s.multiply(Complex {
        re: -phase12_s,
        im: 0.0,
    });
    let phase_p = phase23_p.multiply(Complex {
        re: -phase12_p,
        im: 0.0,
    });
    let reflectance_s = airy_summation(channel, r12_s, r23_s, opd, phase_s);
    let reflectance_p = airy_summation(channel, r12_p, r23_p, opd, phase_p);
    (
        saturate(f64::midpoint(reflectance_s, reflectance_p)),
        cos_theta_3,
    )
}

/// Squared polarized dielectric reflectances, negative transmitted cosine, and phase signs.
fn dielectric_interface(cosine_incident: f64, eta: f64) -> Option<(f64, f64, f64, f64, f64)> {
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
    cosine_incident: f64,
    ambient_ior: f64,
    substrate_n: f64,
    substrate_k: f64,
    f82: f64,
) -> (f64, f64, Complex, Complex) {
    let eta1 = ambient_ior;
    let eta2 = substrate_n;
    let k2 = substrate_k;
    let eta1_squared = eta1 * eta1;
    let eta2_squared = eta2 * eta2;
    let k2_squared = k2 * k2;
    let two_eta2_k2 = 2.0 * eta2 * k2;
    let t1 = eta2_squared - k2_squared - eta1_squared * (1.0 - cosine_incident * cosine_incident);
    let t2 = (t1 * t1 + two_eta2_k2 * two_eta2_k2).sqrt();
    let u_squared = f64::midpoint(t2, t1).max(0.0);
    let v_squared = f64::midpoint(t2, -t1).max(0.0);
    let u = u_squared.sqrt();
    let v = v_squared.sqrt();

    let n = eta2 / eta1;
    let k_squared = (k2 / eta1).powi(2);
    let f0 = ((n - 1.0).powi(2) + k_squared) / ((n + 1.0).powi(2) + k_squared);
    let b = f82_b(f0, f82);
    let reflectance = f82_fresnel_at(cosine_incident, f0, b);

    let re_s = -u_squared - v_squared + (eta1 * cosine_incident).powi(2);
    let im_s = -2.0 * eta1 * cosine_incident * v;
    let phase_s = unit_complex(re_s, im_s);
    let re_p = ((eta2_squared + k2_squared) * cosine_incident).powi(2)
        - eta1_squared * (u_squared + v_squared);
    let im_p = 2.0 * eta1 * cosine_incident * (two_eta2_k2 * u - (eta2_squared - k2_squared) * v);
    let phase_p = unit_complex(re_p, im_p);
    (reflectance, reflectance, phase_s, phase_p)
}

fn unit_complex(re: f64, im: f64) -> Complex {
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

fn f82_b(f0: f64, f82: f64) -> f64 {
    let f: f64 = 6.0 / 7.0;
    let f5 = f.powi(5);
    (7.0 / (f5 * f)) * (f0 + (1.0 - f0) * f5 - f82)
}

fn f82_fresnel_at(cosine: f64, f0: f64, b: f64) -> f64 {
    let s = saturate(1.0 - cosine);
    let s5 = s.powi(5);
    saturate(f0 + (1.0 - f0) * s5 - b * cosine * s5 * s)
}

fn airy_summation(channel: usize, r12: f64, r23: f64, opd: f64, phasor: Complex) -> f64 {
    let t121 = 1.0 - r12;
    let r123 = r12 * r23;
    let r123_root = r123.sqrt();
    let rs = t121 * t121 * r23 / (1.0 - r123);
    let mut result = rs + r12;
    let mut coefficient = rs - t121;
    let mut accumulator = phasor;
    for order in 1..4 {
        coefficient *= r123_root;
        let sensitivity = sensitivity(channel, order as f64 * opd);
        result +=
            coefficient * 2.0 * (accumulator.re * sensitivity.re + accumulator.im * sensitivity.im);
        accumulator = accumulator.multiply(phasor);
    }
    result
}

fn sensitivity(channel: usize, opd: f64) -> Complex {
    let x = saturate(TAU * opd / 60_000.0) * 511.0;
    let index = (x as usize).min(511);
    let next = (index + 1).min(511);
    let fraction = x - index as f64;
    let row0 = THIN_FILM_CMF[index];
    let row1 = THIN_FILM_CMF[next];
    Complex {
        re: f64::from(f32::from_bits(row0[channel])) * (1.0 - fraction)
            + f64::from(f32::from_bits(row1[channel])) * fraction,
        im: f64::from(f32::from_bits(row0[channel + 3])) * (1.0 - fraction)
            + f64::from(f32::from_bits(row1[channel + 3])) * fraction,
    }
}

fn saturate(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}
