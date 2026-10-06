//! Seeded Tessendorf-style ocean spectrum and inverse FFT evaluation.

use std::f64::consts::TAU;

use glam::DVec3;
use serde_json::{Map, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{GridParams, Mesh},
    model::Modifier,
};

use super::{
    bool_param, invalid_parameter, number_param, string_param_with_expected, uint_param, validated,
};

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn from_polar(radius: f64, phase: f64) -> Self {
        Self {
            re: radius * phase.cos(),
            im: radius * phase.sin(),
        }
    }

    fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

/// Evaluate either the generated periodic grid or an input mesh displaced by the spectrum.
pub(super) fn ocean(mesh: &Mesh, modifier: &Modifier, _frame: f64) -> Result<Mesh> {
    let geometry_mode = string_param_with_expected(
        modifier,
        "geometry_mode",
        "GENERATE",
        "a Blender enum or string",
    )?;
    if !["GENERATE", "DISPLACE"].contains(&geometry_mode) {
        return Err(invalid_parameter(
            modifier,
            "geometry_mode",
            "GENERATE or DISPLACE",
        ));
    }
    let _render_resolution = resolution_param(modifier, "resolution", 7)?;
    let viewport_resolution = resolution_param(modifier, "viewport_resolution", 7)?;
    let geometry_resolution = usize::try_from(viewport_resolution)
        .map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "ocean resolution exceeds mesh limits",
            )
        })?
        .checked_mul(usize::try_from(viewport_resolution).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "ocean resolution exceeds mesh limits",
            )
        })?)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "ocean resolution exceeds mesh limits",
            )
        })?;
    let n = geometry_resolution;
    let spatial_size = number_param(modifier, "spatial_size", 50.0)?;
    let size = number_param(modifier, "size", 1.0)?;
    let depth = number_param(modifier, "depth", 200.0)?;
    let wave_scale = number_param(modifier, "wave_scale", 1.0)?;
    let wave_scale_min = number_param(modifier, "wave_scale_min", 0.01)?;
    let choppiness = number_param(modifier, "choppiness", 1.0)?;
    let wind_velocity = number_param(modifier, "wind_velocity", 30.0)?;
    let wave_alignment = number_param(modifier, "wave_alignment", 0.0)?;
    let wave_direction = number_param(modifier, "wave_direction", 0.0)?;
    let damping = number_param(modifier, "damping", 0.5)?;
    let frame_time = number_param(modifier, "time", 1.0)?;
    let repeat_x = repeat_param(modifier, "repeat_x", 1)?;
    let repeat_y = repeat_param(modifier, "repeat_y", 1)?;
    if !spatial_size.is_finite() || spatial_size <= 0.0 {
        return Err(invalid_parameter(
            modifier,
            "spatial_size",
            "a finite positive length in object space",
        ));
    }
    if !size.is_finite() || size <= 0.0 {
        return Err(invalid_parameter(
            modifier,
            "size",
            "a finite positive object-space scale",
        ));
    }
    if !depth.is_finite() || depth < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "depth",
            "a finite non-negative depth",
        ));
    }
    if wave_scale < 0.0 || wave_scale_min < 0.0 || choppiness < 0.0 || wind_velocity <= 0.0 {
        return Err(invalid_parameter(
            modifier,
            "wave_scale",
            "non-negative wave scales, choppiness, and positive wind velocity",
        ));
    }
    if !(0.0..=1.0).contains(&wave_alignment) || !wave_direction.is_finite() {
        return Err(invalid_parameter(
            modifier,
            "wave_alignment",
            "wave_alignment in [0, 1] and a finite wave_direction in radians",
        ));
    }
    if !(0.0..=1.0).contains(&damping) {
        return Err(invalid_parameter(modifier, "damping", "a number in [0, 1]"));
    }
    if !frame_time.is_finite() {
        return Err(invalid_parameter(
            modifier,
            "time",
            "a finite time in seconds",
        ));
    }
    let spectrum =
        string_param_with_expected(modifier, "spectrum", "PHILLIPS", "a Blender enum or string")?;
    if ![
        "PHILLIPS",
        "PIERSON_MOSKOWITZ",
        "JONSWAP",
        "TEXEL_MARSEN_ARSLOE",
    ]
    .contains(&spectrum)
    {
        return Err(invalid_parameter(
            modifier,
            "spectrum",
            "PHILLIPS, PIERSON_MOSKOWITZ, JONSWAP, or TEXEL_MARSEN_ARSLOE",
        ));
    }
    let sharpen_peak = number_param(modifier, "sharpen_peak_jonswap", 0.0)?;
    let fetch = number_param(modifier, "fetch_jonswap", 120.0)?;
    if !(0.0..=1.0).contains(&sharpen_peak) || !fetch.is_finite() || fetch < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "sharpen_peak_jonswap",
            "sharpen_peak_jonswap in [0, 1] and non-negative fetch_jonswap",
        ));
    }
    let use_normals = bool_param(modifier, "use_normals", false)?;
    let use_foam = bool_param(modifier, "use_foam", false)?;
    let foam_layer_name =
        string_param_with_expected(modifier, "foam_layer_name", "", "a Blender enum or string")?;
    let foam_coverage = number_param(modifier, "foam_coverage", 0.0)?;
    let random_seed = seed_param(modifier, "random_seed", 0)?;
    let (spectrum_values, displacement_x, displacement_y) = height_spectrum(
        n,
        spatial_size,
        depth,
        wind_velocity,
        wave_scale,
        wave_scale_min,
        choppiness,
        wave_alignment,
        wave_direction,
        damping,
        frame_time,
        spectrum,
        sharpen_peak,
        fetch,
        random_seed,
    );
    let extent_x = spatial_size * size * f64::from(repeat_x);
    let extent_y = spatial_size * size * f64::from(repeat_y);
    let mut output = if geometry_mode == "GENERATE" {
        let subdivisions = |repeats: u32| {
            geometry_resolution
                .checked_mul(usize::try_from(repeats).map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "ocean repeat count exceeds mesh limits",
                    )
                })?)
                .and_then(|intervals| intervals.checked_add(1))
                .and_then(|vertices| u32::try_from(vertices).ok())
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "ocean grid resolution exceeds mesh limits",
                    )
                })
        };
        let x_subdivisions = subdivisions(repeat_x)?;
        let y_subdivisions = subdivisions(repeat_y)?;
        Mesh::grid(GridParams {
            size_x: extent_x,
            size_y: extent_y,
            x_subdivisions,
            y_subdivisions,
        })
        .map_err(|error| {
            PotError::invalid_argument(format!("ocean grid construction failed: {error}"))
        })?
    } else {
        mesh.clone()
    };
    let spacing_x = spatial_size * size / n as f64;
    let spacing_y = spatial_size * size / n as f64;
    let mut normals = Map::new();
    let mut foam = Map::new();
    for vertex in &mut output.vertices {
        // Blender maps object coordinates to the periodic ocean domain [0, 1].
        let u = ((vertex.co.x / (spatial_size * size) + 0.5) * n as f64).rem_euclid(n as f64);
        let v = ((vertex.co.y / (spatial_size * size) + 0.5) * n as f64).rem_euclid(n as f64);
        let height = sample_grid(&spectrum_values, n, u, v);
        let derivative_x = (sample_grid(&spectrum_values, n, u + 1.0, v)
            - sample_grid(&spectrum_values, n, u - 1.0, v))
            * 0.5;
        let derivative_y = (sample_grid(&spectrum_values, n, u, v + 1.0)
            - sample_grid(&spectrum_values, n, u, v - 1.0))
            * 0.5;
        let displacement_x = sample_grid(&displacement_x, n, u, v);
        let displacement_y = sample_grid(&displacement_y, n, u, v);
        let gradient = DVec3::new(-derivative_x / spacing_x, -derivative_y / spacing_y, 1.0);
        if geometry_mode == "GENERATE" {
            vertex.co.x += displacement_x;
            vertex.co.y += displacement_y;
            vertex.co.z = height;
        } else {
            vertex.co.x += displacement_x;
            vertex.co.y += displacement_y;
            vertex.co.z += height;
        }
        if use_normals {
            let normal = gradient.normalize_or_zero();
            normals.insert(vertex.id.to_string(), json!(normal.to_array()));
        }
        if use_foam {
            let steepness = derivative_x.hypot(derivative_y);
            let foam_amount =
                ((steepness - (1.0 - foam_coverage)) / foam_coverage.max(1.0e-9)).clamp(0.0, 1.0);
            foam.insert(vertex.id.to_string(), json!(foam_amount));
        }
    }
    if use_normals {
        output.attributes.insert(
            "ocean_normal".to_owned(),
            json!({"domain":"vertices","type":"float3","values":normals}),
        );
    }
    if use_foam && !foam_layer_name.is_empty() {
        output.attributes.insert(
            foam_layer_name.to_owned(),
            json!({"domain":"vertices","type":"float","values":foam}),
        );
    }
    validated(&output)?;
    Ok(output)
}

#[expect(
    clippy::too_many_arguments,
    reason = "ocean spectrum consumes the named Blender RNA properties"
)]
fn height_spectrum(
    n: usize,
    spatial_size: f64,
    depth: f64,
    wind_velocity: f64,
    wave_scale: f64,
    wave_scale_min: f64,
    choppiness: f64,
    alignment: f64,
    direction: f64,
    damping: f64,
    time: f64,
    spectrum: &str,
    jonswap_peak: f64,
    fetch: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut frequency = vec![Complex::default(); n * n];
    let mut frequency_x = vec![Complex::default(); n * n];
    let mut frequency_y = vec![Complex::default(); n * n];
    let mut normalization_frequency = vec![Complex::default(); n * n];
    let wind_x = direction.cos();
    let wind_y = -direction.sin();
    let gravity = 9.81;
    let largest_wave = wind_velocity * wind_velocity / gravity;
    for x in 0..n {
        let signed_x = if x <= n / 2 {
            x as f64
        } else {
            x as f64 - n as f64 + 1.0
        };
        let kx = TAU * signed_x / spatial_size;
        let blender_kx = (2.0_f32 * std::f32::consts::PI * signed_x as f32) / spatial_size as f32;
        for y in 0..=n / 2 {
            let ky = TAU * y as f64 / spatial_size;
            let blender_ky = (2.0_f32 * std::f32::consts::PI * y as f32) / spatial_size as f32;
            let wave_number = kx.hypot(ky);
            if wave_number <= 1.0e-12 {
                continue;
            }
            let direction_cosine = (kx * wind_x + ky * wind_y) / wave_number;
            let direction_cosine_reverse = -direction_cosine;
            let adjusted_direction = if direction_cosine < 0.0 {
                direction_cosine * (1.0 - damping)
            } else {
                direction_cosine
            };
            let adjusted_reverse_direction = if direction_cosine_reverse < 0.0 {
                direction_cosine_reverse * (1.0 - damping)
            } else {
                direction_cosine_reverse
            };
            let alignment_factor = adjusted_direction.abs().powf(alignment * 10.0);
            let alignment_factor_reverse = adjusted_reverse_direction.abs().powf(alignment * 10.0);
            let base = (-(1.0 / (wave_number * largest_wave).powi(2))).exp() / wave_number.powi(4)
                * alignment_factor
                * (-(wave_scale_min * wave_scale_min * wave_number.powi(2))).exp();
            let base_reverse = (-(1.0 / (wave_number * largest_wave).powi(2))).exp()
                / wave_number.powi(4)
                * alignment_factor_reverse
                * (-(wave_scale_min * wave_scale_min * wave_number.powi(2))).exp();
            let dispersion = (gravity * wave_number * (wave_number * depth).tanh()).sqrt();
            let spectral_scale = spectrum_factor(
                spectrum,
                kx,
                ky,
                wave_number,
                depth,
                wind_velocity,
                fetch,
                jonswap_peak,
                alignment,
                direction,
                damping,
            );
            let reverse_spectral_scale = spectrum_factor(
                spectrum,
                -kx,
                -ky,
                wave_number,
                depth,
                wind_velocity,
                fetch,
                jonswap_peak,
                alignment,
                direction,
                damping,
            );
            let hash_x = (blender_kx * 360.0) as i32 as u32;
            let hash_y = (blender_ky * 360.0) as i32 as u32;
            let random_seed = (seed as u32).wrapping_add(blender_hash_int_2d(hash_x, hash_y));
            let mut random = BlenderRng::new(random_seed);
            let (real, imaginary) = random.gaussian_pair();
            let gaussian = Complex {
                re: f64::from(real),
                im: f64::from(imaginary),
            };
            let h0 = gaussian.scale((base * spectral_scale * 0.5).max(0.0).sqrt());
            let h0_minus = gaussian.scale(
                (base_reverse * reverse_spectral_scale * 0.5)
                    .max(0.0)
                    .sqrt(),
            );
            let phase = dispersion * time;
            let positive = Complex::from_polar(1.0, phase);
            let negative = Complex::from_polar(1.0, -phase);
            let h0_minus_conjugate = h0_minus.conjugate();
            let height = h0 * positive + h0_minus_conjugate * negative;
            let index = x * n + y;
            frequency[index] = height;
            normalization_frequency[index] = h0 + h0_minus_conjugate;
            let chop = choppiness / wave_number;
            frequency_x[index] = height
                * Complex {
                    re: 0.0,
                    im: chop * kx,
                };
            frequency_y[index] = height
                * Complex {
                    re: 0.0,
                    im: chop * ky,
                };
        }
    }
    for x in 0..n {
        let opposite_x = (n - x) % n;
        for y in n / 2 + 1..n {
            let index = x * n + y;
            let opposite = opposite_x * n + (n - y);
            frequency[index] = frequency[opposite].conjugate();
            frequency_x[index] = frequency_x[opposite].conjugate();
            frequency_y[index] = frequency_y[opposite].conjugate();
            normalization_frequency[index] = normalization_frequency[opposite].conjugate();
        }
    }
    inverse_fft_2d(&mut frequency, n);
    inverse_fft_2d(&mut frequency_x, n);
    inverse_fft_2d(&mut frequency_y, n);
    inverse_fft_2d(&mut normalization_frequency, n);
    let maximum_initial_height = normalization_frequency
        .iter()
        .map(|height| height.re.abs())
        .fold(0.0_f64, f64::max);
    let normalization = 1.0 / maximum_initial_height.max(1.0e-5);
    (
        frequency
            .into_iter()
            .map(|height| height.re * normalization * wave_scale)
            .collect(),
        frequency_x
            .into_iter()
            .map(|displacement| displacement.re * normalization * wave_scale)
            .collect(),
        frequency_y
            .into_iter()
            .map(|displacement| displacement.re * normalization * wave_scale)
            .collect(),
    )
}

impl Complex {
    fn scale(self, amount: f64) -> Self {
        Self {
            re: self.re * amount,
            im: self.im * amount,
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the Blender ocean spectrum depends on named modifier properties"
)]
fn spectrum_factor(
    spectrum: &str,
    kx: f64,
    ky: f64,
    wave_number: f64,
    depth: f64,
    wind: f64,
    fetch: f64,
    sharpen_peak: f64,
    alignment: f64,
    direction: f64,
    damping: f64,
) -> f64 {
    let gravity = 9.81;
    let omega = (gravity * wave_number * (wave_number * depth).tanh()).sqrt();
    let wind_dot = (kx * direction.cos() - ky * direction.sin()) / wave_number.powi(2);
    let apply_wind = |value: f64| {
        let mut weighted = value * wind_dot.abs().powf(alignment * 10.0);
        if wind_dot < 0.0 && alignment > 0.0 {
            weighted *= 1.0 - damping;
        }
        weighted
    };
    let jonswap = || {
        let dimensionless_fetch = (gravity * fetch / wind.sqrt()).abs();
        let alpha = 0.076 * dimensionless_fetch.powf(-0.22);
        let peak_omega =
            std::f64::consts::TAU * 3.5 * (gravity / wind).abs() * dimensionless_fetch.powf(-0.33);
        let gamma = (sharpen_peak * 10.0).clamp(1.0, 6.0);
        let base =
            alpha * gravity.sqrt() / omega.powi(5) * (-1.25 * (peak_omega / omega).powi(4)).exp();
        let sigma: f64 = if omega < peak_omega { 0.07 } else { 0.09 };
        let peak_exponent = -((omega - peak_omega) / (sigma * peak_omega)).powi(2) / 2.0;
        base * gamma.powf(peak_exponent.exp())
    };
    match spectrum {
        "PIERSON_MOSKOWITZ" => {
            let peak_omega = 0.87 * gravity / wind;
            let base = 0.0081 * gravity.sqrt() / omega.powi(5)
                * (-1.291 * (peak_omega / omega).powi(4)).exp();
            apply_wind(base)
        }
        "JONSWAP" => apply_wind(jonswap()),
        "TEXEL_MARSEN_ARSLOE" => {
            let k_depth = omega * (depth / gravity).sqrt();
            let depth_factor = 0.5 + 0.5 * (1.8 * (k_depth - 1.125)).tanh();
            apply_wind(apply_wind(jonswap()) * depth_factor)
        }
        _ => 1.0,
    }
}

fn inverse_fft_2d(values: &mut [Complex], n: usize) {
    for row in values.chunks_exact_mut(n) {
        fft_1d(row, true);
    }
    let mut column = vec![Complex::default(); n];
    for y in 0..n {
        for x in 0..n {
            column[x] = values[x * n + y];
        }
        fft_1d(&mut column, true);
        for x in 0..n {
            values[x * n + y] = column[x];
        }
    }
}

fn fft_1d(values: &mut [Complex], inverse: bool) {
    let n = values.len();
    if !n.is_power_of_two() {
        let mut transformed = [Complex::default(); 81];
        assert!(
            n <= transformed.len(),
            "Ocean FFT exceeds validated resolution"
        );
        let direction = if inverse { 1.0 } else { -1.0 };
        let normalization = if inverse { 1.0 / n as f64 } else { 1.0 };
        for (output_index, output) in transformed.iter_mut().take(n).enumerate() {
            let mut total = Complex::default();
            for (input_index, input) in values.iter().enumerate() {
                let phase = direction * TAU * (input_index * output_index) as f64 / n as f64;
                total = total + *input * Complex::from_polar(1.0, phase);
            }
            *output = total.scale(normalization);
        }
        values.copy_from_slice(&transformed[..n]);
        return;
    }
    let mut reversed = 0_usize;
    for index in 1..n {
        let mut bit = n >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            values.swap(index, reversed);
        }
    }
    let direction = if inverse { 1.0 } else { -1.0 };
    let mut length = 2;
    while length <= n {
        let root = Complex::from_polar(1.0, direction * TAU / length as f64);
        for start in (0..n).step_by(length) {
            let mut twiddle = Complex::from_polar(1.0, 0.0);
            let half = length / 2;
            for offset in 0..half {
                let even = values[start + offset];
                let odd = values[start + offset + half] * twiddle;
                values[start + offset] = even + odd;
                values[start + offset + half] = Complex {
                    re: even.re - odd.re,
                    im: even.im - odd.im,
                };
                twiddle = twiddle * root;
            }
        }
        length <<= 1;
    }
    if inverse {
        for value in values {
            *value = value.scale(1.0 / n as f64);
        }
    }
}

fn sample_grid(values: &[f64], n: usize, x: f64, y: f64) -> f64 {
    let x = x.rem_euclid(n as f64);
    let y = y.rem_euclid(n as f64);
    let x0 = x.floor() as usize % n;
    let y0 = y.floor() as usize % n;
    let x1 = (x0 + 1) % n;
    let y1 = (y0 + 1) % n;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let first = values[x0 * n + y0] * (1.0 - fy) + values[x0 * n + y1] * fy;
    let second = values[x1 * n + y0] * (1.0 - fy) + values[x1 * n + y1] * fy;
    first * (1.0 - fx) + second * fx
}

fn resolution_param(modifier: &Modifier, name: &str, default: u64) -> Result<u32> {
    let value = uint_param(modifier, name, default)?;
    if value == 0 {
        return Err(invalid_parameter(
            modifier,
            name,
            "an integer power-of-two exponent of at least 1",
        ));
    }
    if value > 9 {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            format!(
                "Ocean resolution `{name}` exceeds the configured FFT budget (maximum exponent 9)"
            ),
        ));
    }
    u32::try_from(value).map_err(|_| {
        invalid_parameter(
            modifier,
            name,
            "an integer power-of-two exponent from 1 to 9",
        )
    })
}

fn repeat_param(modifier: &Modifier, name: &str, default: u64) -> Result<u32> {
    let value = uint_param(modifier, name, default)?;
    if !(1..=1024).contains(&value) {
        return Err(invalid_parameter(
            modifier,
            name,
            "an integer from 1 to 1024",
        ));
    }
    u32::try_from(value).map_err(|_| invalid_parameter(modifier, name, "an integer from 1 to 1024"))
}

fn seed_param(modifier: &Modifier, name: &str, default: u64) -> Result<u64> {
    match modifier.params.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .filter(|seed| *seed <= 2_147_483_647)
            .ok_or_else(|| invalid_parameter(modifier, name, "an integer from 0 to 2147483647")),
    }
}

fn blender_hash_int_2d(x: u32, y: u32) -> u32 {
    let initial = 0xdead_beef_u32.wrapping_add((2_u32 << 2) + 13);
    let mut a = initial.wrapping_add(x);
    let mut b = initial.wrapping_add(y);
    let mut c = initial;
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

struct BlenderRng(u64);

impl BlenderRng {
    fn new(seed: u32) -> Self {
        Self((u64::from(seed) << 16) | 0x330e)
    }

    fn get_float(&mut self) -> f32 {
        self.0 =
            (0x0005_deec_e66d_u64.wrapping_mul(self.0).wrapping_add(0xb)) & 0x0000_ffff_ffff_ffff;
        (self.0 >> 17) as f32 / 2_147_483_648.0
    }

    fn gaussian(&mut self) -> f32 {
        loop {
            let x = 1.0 - 2.0 * self.get_float();
            let y = 1.0 - 2.0 * self.get_float();
            let length_squared = x * x + y * y;
            if length_squared > 0.0 && length_squared < 1.0 {
                return x * (-2.0 * length_squared.ln() / length_squared).sqrt();
            }
        }
    }

    fn gaussian_pair(&mut self) -> (f32, f32) {
        (self.gaussian(), self.gaussian())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        geom::{GridParams, Mesh},
        model::{Id, Modifier},
    };
    use serde_json::{Map, json};

    use super::ocean;

    #[test]
    fn spectrum_is_seeded_and_choppiness_changes_horizontal_positions()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = Mesh::grid(GridParams {
            size_x: 8.0,
            size_y: 8.0,
            x_subdivisions: 9,
            y_subdivisions: 9,
        })?;
        let create = |choppiness| -> Result<Modifier, Box<dyn std::error::Error>> {
            Ok(Modifier {
                id: Id::new("ocean_test")?,
                modifier_type: "ocean".to_owned(),
                name: "Ocean".to_owned(),
                enabled: true,
                params: serde_json::from_value::<Map<String, serde_json::Value>>(json!({
                    "geometry_mode":"DISPLACE", "resolution":4, "spatial_size":8.0,
                    "random_seed":12, "choppiness":choppiness, "time":2.0
                }))?,
                binding_data: None,
                runtime: crate::model::ModifierRuntime::default(),
            })
        };
        let first = ocean(&source, &create(0.0)?, 1.0)?;
        let repeated = ocean(&source, &create(0.0)?, 1.0)?;
        assert_eq!(first, repeated);
        let chopped = ocean(&source, &create(2.0)?, 1.0)?;
        assert!(
            first
                .vertices
                .iter()
                .zip(&chopped.vertices)
                .any(|(left, right)| {
                    (left.co.x - right.co.x).abs() + (left.co.y - right.co.y).abs() > 1.0e-8
                })
        );
        assert!(
            first
                .vertices
                .iter()
                .any(|vertex| vertex.co.z.abs() > 1.0e-10)
        );
        Ok(())
    }
}
