// SPDX-License-Identifier: GPL-3.0-or-later
//! Blender 5.2 ocean spectra, frequency layout, and per-wave random seeding.
//!
//! Arithmetic follows `blenkernel/intern/ocean.cc` and `ocean_spectrum.cc`: spectrum terms are
//! single precision, while `pow`/`exp` calls that take double literals in Blender are evaluated in
//! double precision here as well.

use std::f32::consts::PI;

/// `GRAVITY` in `ocean_intern.h`.
pub(super) const GRAVITY: f32 = 9.81;

/// Spectrum selected by the `spectrum` enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Spectrum {
    Phillips,
    PiersonMoskowitz,
    Jonswap,
    TexelMarsenArsloe,
}

impl Spectrum {
    pub(super) fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "PHILLIPS" => Self::Phillips,
            "PIERSON_MOSKOWITZ" => Self::PiersonMoskowitz,
            "JONSWAP" => Self::Jonswap,
            "TEXEL_MARSEN_ARSLOE" => Self::TexelMarsenArsloe,
            _ => return None,
        })
    }
}

/// Modifier properties that feed `BKE_ocean_init`, converted to the single-precision values the
/// Blender ocean stores.
#[derive(Clone, Copy, Debug)]
pub(super) struct Settings {
    pub(super) spectrum: Spectrum,
    /// `_Lx` / `_Lz`: `spatial_size`.
    pub(super) length: f32,
    /// `_V`: `wind_velocity`.
    pub(super) wind_velocity: f32,
    /// `_l`: `wave_scale_min`.
    pub(super) smallest_wave: f32,
    /// `_wx`, `_wz`: wave direction vector.
    pub(super) wind_x: f32,
    pub(super) wind_z: f32,
    /// `_L`: largest wave for `V`.
    pub(super) largest_wave: f32,
    /// `_damp_reflections`: `1 - damping`.
    pub(super) damp_reflections: f32,
    /// `_wind_alignment`: `wave_alignment * 10`.
    pub(super) wind_alignment: f32,
    pub(super) depth: f32,
    /// `_fetch_jonswap`.
    pub(super) fetch: f32,
    /// `_sharpen_peak_jonswap`: `sharpen_peak_jonswap * 10`, clamped to `[1, 6]` at use.
    pub(super) sharpen_peak: f32,
}

impl Settings {
    /// Blender's `BKE_ocean_init` values, read from the modifier's parameters. `length` is `spatial_size`
    /// and `smallest_wave` is `wave_scale_min`; the wind direction, damping, alignment and sharpen peak are
    /// converted as Blender's `ocean_spectrum.cc` uses them.
    pub(super) fn new(params: &super::OceanParams) -> Self {
        Self {
            spectrum: params.spectrum,
            length: params.spatial_size,
            wind_velocity: params.wind_velocity,
            smallest_wave: params.wave_scale_min,
            wind_x: params.wave_direction.cos(),
            wind_z: -params.wave_direction.sin(),
            largest_wave: params.wind_velocity * params.wind_velocity / GRAVITY,
            damp_reflections: 1.0 - params.damping,
            wind_alignment: params.wave_alignment * 10.0,
            depth: params.depth,
            fetch: params.fetch,
            sharpen_peak: params.sharpen_peak * 10.0,
        }
    }
}

/// `BLI_hash_int_2d` (Jenkins lookup3 with two words).
pub(super) fn hash_int_2d(x: u32, y: u32) -> u32 {
    let mut a = 0xdead_beef_u32.wrapping_add((2 << 2) + 13).wrapping_add(x);
    let mut b = 0xdead_beef_u32.wrapping_add((2 << 2) + 13).wrapping_add(y);
    let mut c = 0xdead_beef_u32.wrapping_add((2 << 2) + 13);
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(14));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(11));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(25));
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(16));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(4));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(14));
    c ^= b;
    c.wrapping_sub(b.rotate_left(24))
}

/// Wave number along one axis, indexed as Blender stores it.
///
/// Blender writes `2 pi i / L` for `i <= n / 2` and then `-2 pi ii / L` counting down from the top
/// index, so index `n - 1` maps to zero and the negative band is shifted by one from a true DFT.
pub(super) fn wave_numbers(n: usize, length: f32) -> Vec<f32> {
    let mut values = vec![0.0_f32; n];
    for (index, value) in values.iter_mut().enumerate().take(n / 2 + 1) {
        *value = 2.0 * PI * index as f32 / length;
    }
    let mut counter = 0_usize;
    let mut index = n - 1;
    while index > n / 2 {
        values[index] = -2.0 * PI * counter as f32 / length;
        index -= 1;
        counter += 1;
    }
    values
}

/// Modified Phillips spectrum (`Ph` in `ocean.cc`) with amplitude `A = 1`.
fn phillips(settings: &Settings, kx: f32, kz: f32) -> f32 {
    let k2 = kx * kx + kz * kz;
    if k2 == 0.0 {
        return 0.0;
    }
    let mut tmp = (settings.wind_x * kx + settings.wind_z * kz) / k2.sqrt();
    if tmp < 0.0 {
        tmp *= settings.damp_reflections;
    }
    (-1.0 / (k2 * (settings.largest_wave * settings.largest_wave))).exp()
        * (-k2 * (settings.smallest_wave * settings.smallest_wave)).exp()
        * tmp.abs().powf(settings.wind_alignment)
        / (k2 * k2)
}

/// `alpha_beta_spectrum` from `ocean_spectrum.cc`.
fn alpha_beta(alpha: f32, beta: f32, gamma: f32, omega: f32, peak_omega: f32) -> f32 {
    let scale = f64::from(alpha * gamma.sqrt()) / f64::from(omega).powf(5.0);
    let ratio = f64::from(peak_omega / omega);
    (scale * (-f64::from(beta) * ratio.powf(4.0)).exp()) as f32
}

/// `peak_sharpen` from `ocean_spectrum.cc`.
fn peak_sharpen(omega: f32, peak_omega: f32, gamma: f32) -> f32 {
    let sigma: f32 = if omega < peak_omega { 0.07 } else { 0.09 };
    let quotient = (omega - peak_omega) / (sigma * peak_omega);
    let exponent = -f64::from(quotient * quotient) / 2.0;
    f64::from(gamma).powf(exponent.exp()) as f32
}

/// `ocean_spectrum_wind_and_damp`: wind alignment bias and reflected-wave damping.
fn wind_and_damp(settings: &Settings, kx: f32, kz: f32, value: f32) -> f32 {
    let k2 = kx * kx + kz * kz;
    let k_mag_inv = 1.0 / k2;
    let k_dot_w = (kx * k_mag_inv * settings.wind_x) + (kz * k_mag_inv * settings.wind_z);
    let mut newval = value * k_dot_w.abs().powf(settings.wind_alignment);
    if k_dot_w < 0.0 && settings.wind_alignment > 0.0 {
        newval *= settings.damp_reflections;
    }
    newval
}

/// Dispersion `omega(k) = sqrt(g k tanh(k depth))` in single precision.
pub(super) fn omega(k: f32, depth: f32) -> f32 {
    (GRAVITY * k * (k * depth).tanh()).sqrt()
}

/// `jonswap` from `ocean_spectrum.cc` (without the wind term).
fn jonswap_base(settings: &Settings, k2: f32) -> f32 {
    let k_mag = k2.sqrt();
    let omega = omega(k_mag, settings.depth);
    let gamma = settings.sharpen_peak.clamp(1.0, 6.0);
    let dimensionless_fetch = (GRAVITY * settings.fetch / settings.wind_velocity.sqrt()).abs();
    let alpha = (0.076 * f64::from(dimensionless_fetch).powf(-0.22)) as f32;
    // Blender: `const float m_tau = M_PI * 2`, which is float(2π) before the double multiplication.
    let peak_omega = (f64::from(std::f32::consts::TAU)
        * 3.5
        * f64::from((GRAVITY / settings.wind_velocity).abs())
        * f64::from(dimensionless_fetch).powf(-0.33)) as f32;
    let beta = 1.25;
    alpha_beta(alpha, beta, GRAVITY, omega, peak_omega) * peak_sharpen(omega, peak_omega, gamma)
}

fn pierson_moskowitz(settings: &Settings, kx: f32, kz: f32) -> f32 {
    let k2 = kx * kx + kz * kz;
    if k2 == 0.0 {
        return 0.0;
    }
    let peak_omega = 0.87 * GRAVITY / settings.wind_velocity;
    let omega = omega(k2.sqrt(), settings.depth);
    let value = alpha_beta(0.0081, 1.291, GRAVITY, omega, peak_omega);
    wind_and_damp(settings, kx, kz, value)
}

fn jonswap(settings: &Settings, kx: f32, kz: f32) -> f32 {
    let k2 = kx * kx + kz * kz;
    if k2 == 0.0 {
        return 0.0;
    }
    wind_and_damp(settings, kx, kz, jonswap_base(settings, k2))
}

fn texel_marsen_arsloe(settings: &Settings, kx: f32, kz: f32) -> f32 {
    let k2 = kx * kx + kz * kz;
    if k2 == 0.0 {
        return 0.0;
    }
    let value = wind_and_damp(settings, kx, kz, jonswap_base(settings, k2));
    let gain = (settings.depth / GRAVITY).sqrt();
    let omega = omega(k2.sqrt(), settings.depth);
    // Blender: `kitaigorodskiiDepth_wh` is float, and `- 1.125` is evaluated in double.
    let kitaigorodskii = (0.5 + 0.5 * (1.8 * (f64::from(omega * gain) - 1.125)).tanh()) as f32;
    wind_and_damp(settings, kx, kz, value * kitaigorodskii)
}

/// Spectral energy `S(kx, kz)` for the selected spectrum.
pub(super) fn energy(settings: &Settings, kx: f32, kz: f32) -> f32 {
    match settings.spectrum {
        Spectrum::Phillips => phillips(settings, kx, kz),
        Spectrum::PiersonMoskowitz => pierson_moskowitz(settings, kx, kz),
        Spectrum::Jonswap => jonswap(settings, kx, kz),
        Spectrum::TexelMarsenArsloe => texel_marsen_arsloe(settings, kx, kz),
    }
}

#[cfg(test)]
mod tests {
    use super::{hash_int_2d, wave_numbers};

    #[test]
    fn wave_numbers_follow_blender_index_layout() {
        // For n = 16 index 15 maps to zero and index 9 to -6 * 2 pi / L, not -7.
        let values = wave_numbers(16, 1.0);
        assert_eq!(values[15], 0.0);
        assert!((values[9] + 6.0 * std::f32::consts::TAU).abs() < 1.0e-4);
        assert!((values[8] - 8.0 * std::f32::consts::TAU).abs() < 1.0e-4);
    }
    #[test]
    fn ocean_hash_is_order_sensitive() {
        assert_ne!(hash_int_2d(1, 2), hash_int_2d(2, 1));
    }
}
