// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/modifiers/intern/MOD_ocean.cc, SPDX-FileCopyrightText: Blender Authors, GPL-2.0-or-later.
//   source/blender/blenkernel/intern/ocean.cc, source/blender/blenkernel/intern/ocean_spectrum.cc,
//   source/blender/blenlib/intern/rand.cc, source/blender/blenlib/BLI_rand.hh,
//   source/blender/blenlib/BLI_hash.h, SPDX-License-Identifier: Blender Authors, GPL-2.0-or-later.

//! Native Blender-compatible seeded ocean spectrum evaluation.

mod fft;
mod spectrum;

use std::{collections::BTreeSet, fs, path::Path};

use glam::DVec3;
use serde_json::{Value, json};

use crate::{
    blender_rng::BlenderRng,
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationPurpose, ModifierEvaluationContext},
    geom::{
        Mesh,
        color_layers::{self, ColorDomain, ColorLayer},
        uv_layers::{self, UvLayer},
    },
    model::{Modifier, OceanBakeCache},
};

use super::{
    bool_param, invalid_parameter, number_param, schema_value, string_param_with_expected,
    uint_param, validated,
};
use fft::{C2r2d, Complex};
use spectrum::{Settings as SpectrumSettings, Spectrum, energy, hash_int_2d, omega, wave_numbers};

// Bound the full n-by-n spectrum by the native mesh-element capacity.
const MAX_OCEAN_RESOLUTION: u32 = 1024;
const MAX_OCEAN_SPECTRUM_SAMPLES: usize = 2_000_000;
const MAX_OCEAN_GRID_VERTICES: usize = 2_000_000;

#[derive(Clone)]
struct OceanParams {
    render_resolution: u32,
    viewport_resolution: u32,
    size: f32,
    spatial_size: f32,
    repeat_x: u32,
    repeat_y: u32,
    wave_scale: f32,
    wave_scale_min: f32,
    choppiness: f32,
    wind_velocity: f32,
    wave_alignment: f32,
    wave_direction: f32,
    damping: f32,
    depth: f32,
    time: f32,
    spectrum: Spectrum,
    sharpen_peak: f32,
    fetch: f32,
    random_seed: u32,
    foam_coverage: f32,
    foam_name: String,
    spray_name: String,
    use_foam: bool,
    use_spray: bool,
    use_normals: bool,
    foam_fade: f32,
    invert_spray: bool,
}

impl OceanParams {
    fn read(modifier: &Modifier) -> Result<Self> {
        let geometry_mode =
            string_param_with_expected(modifier, "geometry_mode", "a Blender enum or string")?;
        if !["GENERATE", "DISPLACE"].contains(&geometry_mode) {
            return Err(invalid_parameter(
                modifier,
                "geometry_mode",
                "GENERATE or DISPLACE",
            ));
        }
        let resolution = resolution_param(modifier, "resolution")?;
        let viewport_resolution = resolution_param(modifier, "viewport_resolution")?;
        let size = number_param(modifier, "size")? as f32;
        let spatial_size = number_param(modifier, "spatial_size")? as f32;
        let repeat_x = repeat_param(modifier, "repeat_x")?;
        let repeat_y = repeat_param(modifier, "repeat_y")?;
        let wave_scale = number_param(modifier, "wave_scale")? as f32;
        let wave_scale_min = number_param(modifier, "wave_scale_min")? as f32;
        let choppiness = number_param(modifier, "choppiness")? as f32;
        let wind_velocity = number_param(modifier, "wind_velocity")? as f32;
        let wave_alignment = number_param(modifier, "wave_alignment")? as f32;
        let wave_direction = number_param(modifier, "wave_direction")? as f32;
        let damping = number_param(modifier, "damping")? as f32;
        let depth = number_param(modifier, "depth")? as f32;
        let time = number_param(modifier, "time")? as f32;
        let spectrum_name =
            string_param_with_expected(modifier, "spectrum", "a Blender enum or string")?;
        let spectrum = Spectrum::parse(spectrum_name)
            .ok_or_else(|| invalid_parameter(modifier, "spectrum", "a Blender ocean spectrum"))?;
        let sharpen_peak = number_param(modifier, "sharpen_peak_jonswap")? as f32;
        let fetch = number_param(modifier, "fetch_jonswap")? as f32;
        let random_seed = seed_param(modifier, "random_seed")?;
        let foam_coverage = number_param(modifier, "foam_coverage")? as f32;
        let foam_name =
            string_param_with_expected(modifier, "foam_layer_name", "a Blender enum or string")?
                .to_owned();
        let spray_name =
            string_param_with_expected(modifier, "spray_layer_name", "a Blender enum or string")?
                .to_owned();
        let use_foam = bool_param(modifier, "use_foam")?;
        let use_spray = bool_param(modifier, "use_spray")?;
        let use_normals = bool_param(modifier, "use_normals")?;
        let foam_fade = number_param(modifier, "bake_foam_fade")? as f32;
        let invert_spray = bool_param(modifier, "invert_spray")?;

        if !size.is_finite()
            || !spatial_size.is_finite()
            || !wave_scale.is_finite()
            || !wave_scale_min.is_finite()
            || !choppiness.is_finite()
            || !wind_velocity.is_finite()
            || !wave_alignment.is_finite()
            || !wave_direction.is_finite()
            || !damping.is_finite()
            || !depth.is_finite()
            || !time.is_finite()
            || !sharpen_peak.is_finite()
            || !fetch.is_finite()
            || !foam_coverage.is_finite()
            || !foam_fade.is_finite()
        {
            return Err(invalid_parameter(
                modifier,
                "time",
                "finite ocean parameters",
            ));
        }
        Ok(Self {
            render_resolution: resolution,
            viewport_resolution,
            size,
            spatial_size,
            repeat_x,
            repeat_y,
            wave_scale,
            wave_scale_min,
            choppiness,
            wind_velocity,
            wave_alignment,
            wave_direction,
            damping,
            depth,
            time,
            spectrum,
            sharpen_peak,
            fetch,
            random_seed,
            foam_coverage,
            foam_name,
            spray_name,
            use_foam,
            use_spray,
            use_normals,
            foam_fade,
            invert_spray,
        })
    }

    fn spectrum_settings(&self) -> SpectrumSettings {
        SpectrumSettings::new(
            self.spectrum,
            self.spatial_size,
            self.wind_velocity,
            self.wave_scale_min,
            self.wave_direction,
            self.damping,
            self.wave_alignment,
            self.depth,
            self.fetch,
            self.sharpen_peak,
        )
    }
}

struct OceanSpectrum {
    n: usize,
    kx: Vec<f32>,
    kz: Vec<f32>,
    k: Vec<f32>,
    h0: Vec<Complex>,
    h0_minus: Vec<Complex>,
    fft: C2r2d,
    normalization: f32,
    depth: f32,
}

struct OceanFields {
    n: usize,
    displacement_y: Vec<f64>,
    displacement_x: Vec<f64>,
    displacement_z: Vec<f64>,
    jacobian_xx: Vec<f64>,
    jacobian_zz: Vec<f64>,
    jacobian_xz: Vec<f64>,
    normal_x: Vec<f64>,
    normal_z: Vec<f64>,
    normal_y: f32,
}
pub(crate) struct OceanBakeImage {
    pub(crate) filename: String,
    pub(crate) resolution: usize,
    pub(crate) pixels: Vec<[f32; 4]>,
}
struct OceanCacheFrame {
    width: usize,
    height: usize,
    displacement: Vec<[f32; 4]>,
    foam: Option<Vec<[f32; 4]>>,
    spray: Option<Vec<[f32; 4]>>,
    spray_inverse: Option<Vec<[f32; 4]>>,
}

pub(crate) struct OceanBakeSequence {
    params: OceanParams,
    spectrum: OceanSpectrum,
    previous_foam: Vec<f32>,
}

#[derive(Clone, Copy)]
struct Eigenstuff {
    jminus: f32,
    eplus_x: f32,
    eplus_z: f32,
    eminus_x: f32,
    eminus_z: f32,
}

impl OceanSpectrum {
    fn new(params: &OceanParams, n: usize) -> Result<Self> {
        let count = n.checked_mul(n).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "ocean spectrum exceeds mesh limits",
            )
        })?;
        if count > MAX_OCEAN_SPECTRUM_SAMPLES {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "ocean spectrum exceeds configured sample limit",
            ));
        }
        let settings = params.spectrum_settings();
        let kx = wave_numbers(n, settings.length);
        let kz = wave_numbers(n, settings.length);
        let mut k = vec![0.0; n * (n / 2 + 1)];
        let half = n / 2 + 1;
        for i in 0..n {
            for j in 0..half {
                let kx_i = kx[i];
                let kz_j = kz[j];
                k[i * half + j] = (kx_i * kx_i + kz_j * kz_j).sqrt();
            }
        }
        let mut h0 = vec![Complex::default(); count];
        let mut h0_minus = vec![Complex::default(); count];
        for i in 0..n {
            for j in 0..n {
                let hash_x = (kx[i] * 360.0) as i32 as u32;
                let hash_z = (kz[j] * 360.0) as i32 as u32;
                let seed = params.random_seed.wrapping_add(hash_int_2d(hash_x, hash_z));
                let mut rng = BlenderRng::seed(seed);
                let gaussian = Complex::new(
                    f64::from(blender_gaussian(&mut rng)),
                    f64::from(blender_gaussian(&mut rng)),
                );
                let energy_value = energy(&settings, kx[i], kz[j]);
                let reverse_energy = energy(&settings, -kx[i], -kz[j]);
                h0[i * n + j] = gaussian.scale(f64::from((energy_value / 2.0).sqrt()));
                h0_minus[i * n + j] = gaussian.scale(f64::from((reverse_energy / 2.0).sqrt()));
            }
        }
        let fft = C2r2d::new(n, n);
        let mut spectrum = Self {
            n,
            kx,
            kz,
            k,
            h0,
            h0_minus,
            fft,
            normalization: 1.0,
            depth: settings.depth,
        };
        // Blender's `set_height_normalize_factor`: 1 / max |disp_y| of a time-0, unit-scale simulation.
        let initial = spectrum.compute(0.0, 1.0, 0.0, false, false, false);
        let max_height = initial
            .displacement_y
            .iter()
            .fold(0.0_f64, |max, value| max.max(value.abs()));
        let max_height = if max_height == 0.0 {
            1.0e-5
        } else {
            max_height
        };
        spectrum.normalization = (1.0 / max_height) as f32;
        Ok(spectrum)
    }

    fn compute(
        &self,
        time: f32,
        wave_scale: f32,
        choppiness: f32,
        do_jacobian: bool,
        do_chop: bool,
        do_normals: bool,
    ) -> OceanFields {
        let n = self.n;
        let half = n / 2 + 1;
        let mut htilda = vec![Complex::default(); n * half];
        for i in 0..n {
            for j in 0..half {
                let k = self.k[i * half + j];
                let phase = f64::from(omega(k, self.depth) * time);
                let positive = Complex::from_polar(1.0, phase);
                let negative = Complex::from_polar(1.0, -phase);
                let h0 = self.h0[i * n + j];
                let h0_minus = self.h0_minus[i * n + j].conj();
                htilda[i * half + j] = h0 * positive + h0_minus * negative;
            }
        }
        let scale = f64::from(wave_scale * self.normalization);
        let scaled: Vec<Complex> = htilda.iter().map(|value| value.scale(scale)).collect();
        let displacement_y = self.fft.run(&scaled);
        let count = n * n;
        let mut displacement_x = vec![0.0; count];
        let mut displacement_z = vec![0.0; count];
        if do_chop && choppiness > 0.0 {
            let chop_scale = f64::from(choppiness) * scale;
            let mut chop_x = vec![Complex::default(); n * half];
            let mut chop_z = vec![Complex::default(); n * half];
            for i in 0..n {
                for j in 0..half {
                    let k = self.k[i * half + j];
                    let (ratio_x, ratio_z) = if k == 0.0 {
                        (0.0, 0.0)
                    } else {
                        (f64::from(self.kx[i] / k), f64::from(self.kz[j] / k))
                    };
                    let h = htilda[i * half + j];
                    chop_x[i * half + j] =
                        Complex::new(-h.im * chop_scale * ratio_x, h.re * chop_scale * ratio_x);
                    chop_z[i * half + j] =
                        Complex::new(-h.im * chop_scale * ratio_z, h.re * chop_scale * ratio_z);
                }
            }
            displacement_x = self.fft.run(&chop_x);
            displacement_z = self.fft.run(&chop_z);
        }
        let (normal_x, normal_z, normal_y) = if do_normals {
            let mut nx = vec![Complex::default(); n * half];
            let mut nz = vec![Complex::default(); n * half];
            for i in 0..n {
                for j in 0..half {
                    let height = htilda[i * half + j];
                    nx[i * half + j] = Complex::new(
                        height.im * f64::from(self.kx[i]),
                        -height.re * f64::from(self.kx[i]),
                    );
                    nz[i * half + j] = Complex::new(
                        height.im * f64::from(self.kz[i]),
                        -height.re * f64::from(self.kz[i]),
                    );
                }
            }
            (
                self.fft.run(&nx),
                self.fft.run(&nz),
                1.0 / (wave_scale * self.normalization),
            )
        } else {
            (vec![0.0; count], vec![0.0; count], 0.0)
        };
        let (jacobian_xx, jacobian_zz, jacobian_xz) = if do_jacobian {
            let chop = f64::from(choppiness);
            let mut jxx = vec![Complex::default(); n * half];
            let mut jzz = vec![Complex::default(); n * half];
            let mut jxz = vec![Complex::default(); n * half];
            for i in 0..n {
                for j in 0..half {
                    let k = self.k[i * half + j];
                    let h = htilda[i * half + j];
                    let (xx, zz, xz) = if k == 0.0 {
                        (0.0, 0.0, 0.0)
                    } else {
                        let kx = f64::from(self.kx[i]);
                        let kz = f64::from(self.kz[j]);
                        (
                            kx * kx / f64::from(k),
                            kz * kz / f64::from(k),
                            kx * kz / f64::from(k),
                        )
                    };
                    let multiplier = -chop;
                    jxx[i * half + j] =
                        Complex::new(multiplier * h.re * xx, multiplier * h.im * xx);
                    jzz[i * half + j] =
                        Complex::new(multiplier * h.re * zz, multiplier * h.im * zz);
                    jxz[i * half + j] =
                        Complex::new(multiplier * h.re * xz, multiplier * h.im * xz);
                }
            }
            let mut xx = self.fft.run(&jxx);
            let mut zz = self.fft.run(&jzz);
            let xz = self.fft.run(&jxz);
            for value in &mut xx {
                *value += 1.0;
            }
            for value in &mut zz {
                *value += 1.0;
            }
            (xx, zz, xz)
        } else {
            (vec![1.0; count], vec![1.0; count], vec![0.0; count])
        };
        OceanFields {
            n,
            displacement_y,
            displacement_x,
            displacement_z,
            jacobian_xx,
            jacobian_zz,
            jacobian_xz,
            normal_x,
            normal_z,
            normal_y,
        }
    }
}

impl OceanFields {
    fn sample(&self, values: &[f64], u: f32, v: f32) -> f32 {
        sample_grid(values, self.n, self.n, u, v)
    }

    fn eigenstuff(&self, u: f32, v: f32) -> Eigenstuff {
        eigenstuff_from_jacobian(
            self.sample(&self.jacobian_xx, u, v),
            self.sample(&self.jacobian_zz, u, v),
            self.sample(&self.jacobian_xz, u, v),
        )
    }

    fn eigenstuff_at(&self, index: usize) -> Eigenstuff {
        eigenstuff_from_jacobian(
            self.jacobian_xx[index] as f32,
            self.jacobian_zz[index] as f32,
            self.jacobian_xz[index] as f32,
        )
    }
}

fn eigenstuff_from_jacobian(jxx: f32, jzz: f32, jxz: f32) -> Eigenstuff {
    let a = jxx + jzz;
    let b = ((jxx - jzz) * (jxx - jzz) + 4.0 * jxz * jxz).sqrt();
    let jminus = 0.5 * (a - b);
    let jplus = 0.5 * (a + b);
    let qplus = (jplus - jxx) / jxz;
    let qminus = (jminus - jxx) / jxz;
    let norm_plus = (1.0 + qplus * qplus).sqrt();
    let norm_minus = (1.0 + qminus * qminus).sqrt();
    Eigenstuff {
        jminus,
        eplus_x: 1.0 / norm_plus,
        eplus_z: qplus / norm_plus,
        eminus_x: 1.0 / norm_minus,
        eminus_z: qminus / norm_minus,
    }
}

/// Field index for EXR pixel `pixel` of an `n`-by-`n` bake. The spectrum fields are stored as `i * n + j` with
/// `i` along u, but a bake row is v-major (`row * n + col`, the layout `sample_cache_uv` reads), so
/// `col = i` and `row = j`. Blender writes its bake in this row-major order.
fn image_field_index(pixel: usize, n: usize) -> usize {
    (pixel % n) * n + pixel / n
}

impl OceanBakeSequence {
    pub(crate) fn new(modifier: &Modifier) -> Result<Self> {
        let params = OceanParams::read(modifier)?;
        let resolution = usize::try_from(params.render_resolution).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "ocean bake resolution exceeds platform limits",
            )
        })?;
        let n = resolution.checked_mul(resolution).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "ocean bake resolution exceeds mesh limits",
            )
        })?;
        let count = n.checked_mul(n).ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "ocean bake exceeds mesh limits")
        })?;
        let spectrum = OceanSpectrum::new(&params, n)?;
        let previous_foam = if params.use_foam {
            vec![0.0; count]
        } else {
            Vec::new()
        };
        Ok(Self {
            params,
            spectrum,
            previous_foam,
        })
    }

    pub(crate) fn bake_frame(&mut self, frame: i32, time: f32) -> Result<Vec<OceanBakeImage>> {
        if !time.is_finite() {
            return Err(PotError::invalid_argument("ocean bake time must be finite"));
        }
        let params = &self.params;
        let fields = self.spectrum.compute(
            time,
            params.wave_scale,
            params.choppiness,
            params.use_foam,
            params.choppiness > 0.0,
            params.use_normals,
        );
        let count = fields.n * fields.n;
        let mut images = Vec::with_capacity(5);
        let mut displacement = Vec::with_capacity(count);
        for pixel in 0..count {
            let index = image_field_index(pixel, fields.n);
            displacement.push([
                fields.displacement_x[index] as f32,
                fields.displacement_y[index] as f32,
                fields.displacement_z[index] as f32,
                1.0,
            ]);
        }
        images.push(OceanBakeImage {
            filename: format!("disp_{frame:04}.exr"),
            resolution: fields.n,
            pixels: displacement,
        });

        if params.use_foam {
            let mut foam = Vec::with_capacity(count);
            let mut spray = params.use_spray.then(|| Vec::with_capacity(count));
            let mut spray_inverse = params.use_spray.then(|| Vec::with_capacity(count));
            for pixel in 0..count {
                let index = image_field_index(pixel, fields.n);
                let eigen = fields.eigenstuff_at(index);
                let fresh = (eigen.jminus * -0.005 + params.foam_coverage).clamp(0.0, 1.0);
                let previous = &mut self.previous_foam[index];
                if *previous < 1.0 {
                    *previous *= *previous;
                }
                let neg_eplus = if eigen.eplus_z < 0.0 {
                    (1.0 + eigen.eplus_z).max(0.0)
                } else {
                    1.0
                };
                *previous *= params.foam_fade * (0.75 + neg_eplus * 0.25);
                *previous = (*previous + fresh).min(1.0);
                foam.push([*previous, *previous, *previous, 1.0]);
                if let Some(spray) = &mut spray {
                    spray.push([eigen.eplus_x, 0.0, eigen.eplus_z, 1.0]);
                }
                if let Some(spray_inverse) = &mut spray_inverse {
                    spray_inverse.push([eigen.eminus_x, 0.0, eigen.eminus_z, 1.0]);
                }
            }
            images.push(OceanBakeImage {
                filename: format!("foam_{frame:04}.exr"),
                resolution: fields.n,
                pixels: foam,
            });
            if let Some(pixels) = spray {
                images.push(OceanBakeImage {
                    filename: format!("spray_{frame:04}.exr"),
                    resolution: fields.n,
                    pixels,
                });
            }
            if let Some(pixels) = spray_inverse {
                images.push(OceanBakeImage {
                    filename: format!("spray_inverse_{frame:04}.exr"),
                    resolution: fields.n,
                    pixels,
                });
            }
        }
        if params.use_normals {
            let mut normals = Vec::with_capacity(count);
            for pixel in 0..count {
                let index = image_field_index(pixel, fields.n);
                let mut normal = [
                    fields.normal_x[index] as f32,
                    fields.normal_y,
                    fields.normal_z[index] as f32,
                ];
                let length =
                    (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                for component in &mut normal {
                    *component /= length;
                }
                normals.push([normal[0], normal[1], normal[2], 1.0]);
            }
            images.push(OceanBakeImage {
                filename: format!("normal_{frame:04}.exr"),
                resolution: fields.n,
                pixels: normals,
            });
        }
        Ok(images)
    }
}

/// Blender's `gaussRand` over the shared `BLI_rand` stream: the polar accept, with the second value discarded.
fn blender_gaussian(rng: &mut BlenderRng) -> f32 {
    loop {
        let x = 1.0 - 2.0 * uniform(rng);
        let y = 1.0 - 2.0 * uniform(rng);
        let length_squared = x * x + y * y;
        if length_squared > 0.0 && length_squared < 1.0 {
            return x * (-2.0 * length_squared.ln() / length_squared).sqrt();
        }
    }
}

/// `BLI_rand`'s float in `[0, 1)`: the 31-bit output over 2^31.
fn uniform(rng: &mut BlenderRng) -> f32 {
    rng.get_uint32() as f32 / 2_147_483_648.0
}

/// Evaluates either Blender's generated periodic grid or an input mesh displaced by the spectrum.
#[cfg(test)]
pub(super) fn ocean(
    mesh: &Mesh,
    modifier: &Modifier,
    frame: f64,
    context: ModifierEvaluationContext<'_>,
) -> Result<Mesh> {
    ocean_with_parameter_evaluator(mesh, modifier, frame, context, |_, _| Ok(None))
}

/// Live evaluation resolves the animated `time` through `evaluate_parameter` once per frame.
pub(super) fn ocean_with_parameter_evaluator(
    mesh: &Mesh,
    modifier: &Modifier,
    frame: f64,
    context: ModifierEvaluationContext<'_>,
    mut evaluate_parameter: impl FnMut(&str, f64) -> Result<Option<f64>>,
) -> Result<Mesh> {
    let params = OceanParams::read(modifier)?;
    let geometry_mode =
        string_param_with_expected(modifier, "geometry_mode", "a Blender enum or string")?;
    let resolution = if context.purpose == EvaluationPurpose::Render {
        params.render_resolution
    } else {
        params.viewport_resolution
    };
    let n = usize::try_from(resolution)
        .ok()
        .and_then(|resolution| resolution.checked_mul(resolution))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "ocean resolution exceeds mesh limits",
            )
        })?;
    let size_co_inv = 1.0_f32 / (params.size * params.spatial_size);
    if !size_co_inv.is_finite() {
        return Ok(mesh.clone());
    }
    let (x_intervals, y_intervals) = if geometry_mode == "GENERATE" {
        let repeat_x = usize::try_from(params.repeat_x).unwrap_or(usize::MAX);
        let repeat_y = usize::try_from(params.repeat_y).unwrap_or(usize::MAX);
        let x_intervals = n.checked_mul(repeat_x).ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "ocean grid exceeds mesh limits")
        })?;
        let y_intervals = n.checked_mul(repeat_y).ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "ocean grid exceeds mesh limits")
        })?;
        let vertex_count = x_intervals
            .checked_add(1)
            .and_then(|x| y_intervals.checked_add(1).and_then(|y| x.checked_mul(y)))
            .ok_or_else(|| {
                PotError::new(ErrorCode::LimitExceeded, "ocean grid exceeds mesh limits")
            })?;
        if vertex_count > MAX_OCEAN_GRID_VERTICES {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "ocean grid exceeds configured vertex limit",
            ));
        }
        (x_intervals, y_intervals)
    } else {
        (0, 0)
    };

    let mut output = if geometry_mode == "GENERATE" {
        generate_ocean_geometry(&params, n, x_intervals, y_intervals)?
    } else {
        mesh.clone()
    };
    let cached = modifier
        .baked_ocean_cache
        .as_ref()
        .map(|cache| load_ocean_cache_frame(modifier, &params, cache, frame as i32, context))
        .transpose()?;
    let fields = if cached.is_none() {
        let base_time = f64::from(params.time);
        let time = evaluate_parameter("time", base_time)?.unwrap_or(base_time) as f32;
        if !time.is_finite() {
            return Err(invalid_parameter(
                modifier,
                "time",
                "a finite Blender float",
            ));
        }
        let spectrum = OceanSpectrum::new(&params, n)?;
        Some(spectrum.compute(
            time,
            params.wave_scale,
            params.choppiness,
            params.use_foam,
            params.choppiness > 0.0,
            params.use_normals,
        ))
    } else {
        None
    };
    let original_positions: Vec<DVec3> = output.vertices.iter().map(|vertex| vertex.co).collect();
    let vertex_index: std::collections::HashMap<u32, usize> = output
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut color_store = color_layers::read(&output);
    let mut used_names = layer_names(&output);
    used_names.extend(color_store.layers.iter().map(|layer| layer.name.clone()));
    let foam_name = params
        .use_foam
        .then(|| unique_attribute_name(&params.foam_name, &mut used_names));
    let spray_name = (params.use_foam && params.use_spray)
        .then(|| unique_attribute_name(&params.spray_name, &mut used_names));

    if let Some(name) = &foam_name {
        let mut values =
            Vec::with_capacity(output.faces.iter().map(|face| face.vertices.len()).sum());
        for face in &output.faces {
            for vertex_id in &face.vertices {
                let position = original_positions[vertex_index[vertex_id]];
                let (u, v) = ocean_uv(position, size_co_inv);
                let foam = if let Some(cache) = &cached {
                    let pixels = cache.foam.as_deref().ok_or_else(|| {
                        PotError::new(ErrorCode::SceneInvalid, "Ocean foam cache image is missing")
                    })?;
                    sample_cache_uv(pixels, cache.width, cache.height, u, v, n, n)[0]
                        .clamp(0.0, 1.0)
                } else {
                    let fields = fields.as_ref().ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "live Ocean spectrum is missing")
                    })?;
                    let eigen = fields.eigenstuff(u, v);
                    (eigen.jminus * -0.005 + params.foam_coverage).clamp(0.0, 1.0)
                };
                let color = byte_color(foam);
                values.push(srgb_color(color, color, color));
            }
        }
        color_store.layers.push(ColorLayer {
            name: name.clone(),
            domain: ColorDomain::Corner,
            values,
        });
        set_byte_color_data_type(&mut output, name)?;
    }
    if let Some(name) = &spray_name {
        let mut values =
            Vec::with_capacity(output.faces.iter().map(|face| face.vertices.len()).sum());
        for face in &output.faces {
            let first = values.len();
            values.resize(first + face.vertices.len(), [1.0, 1.0, 1.0, 1.0]);
            if let Some(vertex_id) = face.vertices.last() {
                let position = original_positions[vertex_index[vertex_id]];
                let (u, v) = ocean_uv(position, size_co_inv);
                let (red, blue) = if let Some(cache) = &cached {
                    let pixels = if params.invert_spray {
                        cache.spray_inverse.as_deref()
                    } else {
                        cache.spray.as_deref()
                    }
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::SceneInvalid,
                            "Ocean spray cache image is missing",
                        )
                    })?;
                    let sample = sample_cache_uv(pixels, cache.width, cache.height, u, v, n, n);
                    (sample[0], sample[2])
                } else {
                    let fields = fields.as_ref().ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "live Ocean spectrum is missing")
                    })?;
                    let eigen = fields.eigenstuff(u, v);
                    if params.invert_spray {
                        (eigen.eminus_x, eigen.eminus_z)
                    } else {
                        (eigen.eplus_x, eigen.eplus_z)
                    }
                };
                values[first] = srgb_color(spray_byte(red), 0, spray_byte(blue));
            }
        }
        color_store.layers.push(ColorLayer {
            name: name.clone(),
            domain: ColorDomain::Corner,
            values,
        });
        set_byte_color_data_type(&mut output, name)?;
    }
    color_layers::write(&mut output, &color_store);

    for (index, vertex) in output.vertices.iter_mut().enumerate() {
        let original = original_positions[index];
        let (u, v) = ocean_uv(original, size_co_inv);
        let (dx, dy, dz) = if let Some(cache) = &cached {
            let displacement =
                sample_cache_uv(&cache.displacement, cache.width, cache.height, u, v, n, n);
            (
                if params.choppiness > 0.0 {
                    displacement[0]
                } else {
                    0.0
                },
                if params.choppiness > 0.0 {
                    displacement[2]
                } else {
                    0.0
                },
                displacement[1],
            )
        } else {
            let fields = fields.as_ref().ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "live Ocean spectrum is missing")
            })?;
            (
                if params.choppiness > 0.0 {
                    fields.sample(&fields.displacement_x, u, v)
                } else {
                    0.0
                },
                if params.choppiness > 0.0 {
                    fields.sample(&fields.displacement_z, u, v)
                } else {
                    0.0
                },
                fields.sample(&fields.displacement_y, u, v),
            )
        };
        vertex.co = DVec3::new(
            f64::from(original.x as f32 + dx),
            f64::from(original.y as f32 + dy),
            f64::from(original.z as f32 + dz),
        );
    }
    validated(&output)?;
    Ok(output)
}

fn generate_ocean_geometry(
    params: &OceanParams,
    n: usize,
    x_intervals: usize,
    y_intervals: usize,
) -> Result<Mesh> {
    let vertex_count = (x_intervals + 1)
        .checked_mul(y_intervals + 1)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "ocean grid exceeds mesh limits"))?;
    let length = params.size * params.spatial_size;
    let origin = -length / 2.0;
    let step = length / n as f32;
    let mut positions = Vec::with_capacity(vertex_count);
    for y in 0..=y_intervals {
        for x in 0..=x_intervals {
            positions.push(DVec3::new(
                f64::from(origin + x as f32 * step),
                f64::from(origin + y as f32 * step),
                0.0,
            ));
        }
    }
    let face_count = x_intervals
        .checked_mul(y_intervals)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "ocean grid exceeds mesh limits"))?;
    let mut polygons = Vec::with_capacity(face_count);
    let mut uv_faces = Vec::with_capacity(face_count);
    let uv_step = 1.0_f32 / n as f32;
    let row_width = x_intervals + 1;
    for y in 0..y_intervals {
        for x in 0..x_intervals {
            let lower_left = y * row_width + x;
            polygons.push(vec![
                lower_left,
                lower_left + 1,
                lower_left + row_width + 1,
                lower_left + row_width,
            ]);
            let u0 = f64::from(x as f32 * uv_step);
            let u1 = f64::from((x + 1) as f32 * uv_step);
            let v0 = f64::from(1.0_f32 - y as f32 * uv_step);
            let v1 = f64::from(1.0_f32 - (y + 1) as f32 * uv_step);
            uv_faces.push(Some(vec![[u0, v0], [u1, v0], [u1, v1], [u0, v1]]));
        }
    }
    let mut output = Mesh::from_positions_and_faces(positions, polygons).map_err(|error| {
        PotError::invalid_argument(format!("ocean grid construction failed: {error}"))
    })?;
    let mut uv_store = uv_layers::read(&output);
    uv_store.layers.push(UvLayer::new("UVMap", uv_faces));
    if uv_store.active.is_none() {
        uv_store.active = Some("UVMap".to_owned());
    }
    if uv_store.render.is_none() {
        uv_store.render = Some("UVMap".to_owned());
    }
    uv_layers::write(&mut output, &uv_store);
    Ok(output)
}

fn ocean_uv(position: DVec3, size_co_inv: f32) -> (f32, f32) {
    (
        (position.x as f32 * size_co_inv) + 0.5,
        (position.y as f32 * size_co_inv) + 0.5,
    )
}

fn sample_grid(values: &[f64], rows: usize, cols: usize, u: f32, v: f32) -> f32 {
    let mut u = u % 1.0;
    let mut v = v % 1.0;
    if u < 0.0 {
        u += 1.0;
    }
    if v < 0.0 {
        v += 1.0;
    }
    let uu = u * rows as f32;
    let vv = v * cols as f32;
    let floor_x = uu.floor();
    let floor_z = vv.floor();
    let i0 = floor_x as usize;
    let j0 = floor_z as usize;
    let i1 = (i0 + 1) % rows;
    let j1 = (j0 + 1) % cols;
    let i0 = i0 % rows;
    let j0 = j0 % cols;
    let fx = uu - floor_x;
    let fz = vv - floor_z;
    let lerp = |target: f64, origin: f64, factor: f32| {
        factor * target as f32 + (1.0 - factor) * origin as f32
    };
    let first = lerp(values[i1 * cols + j1], values[i0 * cols + j1], fx);
    let second = lerp(values[i1 * cols + j0], values[i0 * cols + j0], fx);
    lerp(f64::from(first), f64::from(second), fz)
}
fn sample_cache_uv(
    pixels: &[[f32; 4]],
    width: usize,
    height: usize,
    u: f32,
    v: f32,
    resolution_x: usize,
    resolution_y: usize,
) -> [f32; 4] {
    let mut u = u % 1.0;
    let mut v = v % 1.0;
    if u < 0.0 {
        u += 1.0;
    }
    if v < 0.0 {
        v += 1.0;
    }
    let texel_x = u * resolution_x as f32 - 0.5;
    let texel_y = v * resolution_y as f32 - 0.5;
    let extend = |coordinate: f32, size: usize| {
        if coordinate >= 0.0 {
            if coordinate < size as f32 {
                coordinate as usize
            } else {
                size - 1
            }
        } else {
            0
        }
    };
    let x1 = extend(texel_x, width);
    let x2 = extend(texel_x + 1.0, width);
    let y1 = extend(texel_y, height);
    let y2 = extend(texel_y + 1.0, height);
    let a = texel_x - texel_x.floor();
    let b = texel_y - texel_y.floor();
    let a_b = a * b;
    let ma_b = (1.0 - a) * b;
    let a_mb = a * (1.0 - b);
    let ma_mb = (1.0 - a) * (1.0 - b);
    let row1 = &pixels[width * y1 + x1];
    let row2 = &pixels[width * y2 + x1];
    let row3 = &pixels[width * y1 + x2];
    let row4 = &pixels[width * y2 + x2];
    std::array::from_fn(|channel| {
        ma_mb * row1[channel] + a_mb * row3[channel] + ma_b * row2[channel] + a_b * row4[channel]
    })
}

fn ocean_cache_directory(
    filepath: &str,
    project_root: Option<&Path>,
) -> Result<std::path::PathBuf> {
    if filepath.is_empty() {
        return Err(PotError::invalid_argument("Ocean cache filepath is empty"));
    }
    if let Some(relative) = filepath.strip_prefix("//") {
        return project_root.map(|root| root.join(relative)).ok_or_else(|| {
            PotError::invalid_argument("project root is required for a // Ocean cache filepath")
        });
    }
    let path = Path::new(filepath);
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        project_root.map(|root| root.join(path)).ok_or_else(|| {
            PotError::invalid_argument(
                "project root is required for a relative Ocean cache filepath",
            )
        })
    }
}

fn load_ocean_cache_image(
    directory: &Path,
    filename: &str,
    expected_hash: &str,
    expected_dimension: usize,
) -> Result<Vec<[f32; 4]>> {
    if Path::new(filename).components().count() != 1
        || Path::new(filename)
            .file_name()
            .and_then(|name| name.to_str())
            != Some(filename)
    {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "Ocean cache file name must be a single path component",
        ));
    }
    let path = directory.join(filename);
    let bytes = fs::read(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PotError::with_details(
                ErrorCode::FileNotFound,
                format!("Ocean cache image could not be read: {error}"),
                json!({"path":path}),
            )
        } else {
            PotError::io(&error)
        }
    })?;
    let actual_hash = crate::hash::sha256(&bytes);
    if actual_hash != expected_hash {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "Ocean cache image digest does not match its baked manifest",
            json!({"path":path,"expected":expected_hash,"actual":actual_hash}),
        ));
    }
    let (width, height, pixels) = crate::image::storage::decode_ocean_cache_exr(&bytes)?;
    if width as usize != expected_dimension || height as usize != expected_dimension {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "Ocean cache image dimensions do not match the baked resolution",
            json!({"path":path,"expected":[expected_dimension,expected_dimension],"actual":[width,height]}),
        ));
    }
    Ok(pixels
        .into_iter()
        .map(|pixel| pixel.map(|value| value as f32))
        .collect())
}

fn load_ocean_cache_frame(
    modifier: &Modifier,
    params: &OceanParams,
    cache: &OceanBakeCache,
    frame: i32,
    context: ModifierEvaluationContext<'_>,
) -> Result<OceanCacheFrame> {
    if cache.frame_start > cache.frame_end || cache.resolution == 0 {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "Ocean baked-cache range or resolution is invalid",
        ));
    }
    let image_resolution = usize::try_from(cache.resolution).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "Ocean cache resolution exceeds platform limits",
        )
    })?;
    let sample_count = image_resolution
        .checked_mul(image_resolution)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "Ocean cache sample count exceeds platform limits",
            )
        })?;
    if sample_count > MAX_OCEAN_SPECTRUM_SAMPLES {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "Ocean cache sample count exceeds configured limit",
        ));
    }
    let filepath = string_param_with_expected(modifier, "filepath", "an Ocean cache directory")?;
    let directory = ocean_cache_directory(filepath, context.project_root)?;
    let frame = frame.clamp(cache.frame_start, cache.frame_end);
    let load_channel = |channel: &str, required: bool| -> Result<Option<Vec<[f32; 4]>>> {
        let filename = format!("{channel}_{frame:04}.exr");
        let Some(digest) = cache.manifest.get(&filename) else {
            return if required {
                Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "Ocean baked-cache manifest is missing a required image",
                    json!({"modifier_id":modifier.id,"file":filename}),
                ))
            } else {
                Ok(None)
            };
        };
        load_ocean_cache_image(&directory, &filename, digest, image_resolution).map(Some)
    };
    let displacement = load_channel("disp", true)?.ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "Ocean displacement cache image is missing",
        )
    })?;
    let foam = load_channel("foam", params.use_foam)?;
    let spray_enabled = params.use_foam && params.use_spray;
    let spray = load_channel("spray", spray_enabled)?;
    let spray_inverse = load_channel("spray_inverse", spray_enabled)?;
    let _normal = load_channel("normal", params.use_normals)?;
    Ok(OceanCacheFrame {
        width: image_resolution,
        height: image_resolution,
        displacement,
        foam,
        spray,
        spray_inverse,
    })
}

fn byte_color(value: f32) -> u8 {
    if value.is_nan() {
        return 0;
    }
    (value.clamp(0.0, 1.0) * 255.0) as u8
}

fn spray_byte(value: f32) -> u8 {
    if !value.is_finite() {
        return 0;
    }
    ((value * 255.0) as i32) as u8
}

fn srgb_color(red: u8, green: u8, blue: u8) -> [f64; 4] {
    let decode = |channel| crate::color::srgb_to_linear(f64::from(channel) / 255.0);
    [decode(red), decode(green), decode(blue), 1.0]
}
fn set_byte_color_data_type(mesh: &mut Mesh, name: &str) -> Result<()> {
    let metadata = mesh
        .attributes
        .entry("blender_metadata".to_owned())
        .or_insert_with(|| json!({}));
    let metadata = metadata.as_object_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::ValidationFailed,
            "Blender mesh metadata is not an object",
        )
    })?;
    let data_types = metadata
        .entry("color_data_types".to_owned())
        .or_insert_with(|| json!({}));
    let data_types = data_types.as_object_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::ValidationFailed,
            "Blender color data types are not an object",
        )
    })?;
    data_types.insert(name.to_owned(), json!("BYTE_COLOR"));
    Ok(())
}

fn unique_attribute_name(requested: &str, used: &mut BTreeSet<String>) -> String {
    let base = if requested.is_empty() {
        "Attribute"
    } else {
        requested
    };
    if used.insert(base.to_owned()) {
        return base.to_owned();
    }
    for suffix in 1_u32.. {
        let name = format!("{base}.{suffix:03}");
        if used.insert(name.clone()) {
            return name;
        }
    }
    unreachable!("the attribute namespace cannot contain every u32 suffix")
}
fn layer_names(mesh: &Mesh) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = mesh
        .attributes
        .keys()
        .filter_map(|key| crate::geom::attribute_keys::user_name(key).map(str::to_owned))
        .collect();
    names.extend(
        uv_layers::read(mesh)
            .layers
            .into_iter()
            .map(|layer| layer.name),
    );
    for (key, field) in [
        ("blender_uv_layers", "name"),
        ("blender_attributes", "name"),
    ] {
        if let Some(layers) = mesh.attributes.get(key).and_then(Value::as_array) {
            names.extend(
                layers
                    .iter()
                    .filter_map(|layer| layer.get(field).and_then(Value::as_str))
                    .map(str::to_owned),
            );
        }
    }
    names
}

fn resolution_param(modifier: &Modifier, name: &str) -> Result<u32> {
    let value = uint_param(modifier, name)?;
    if !(1..=MAX_OCEAN_RESOLUTION as u64).contains(&value) {
        return Err(invalid_parameter(
            modifier,
            name,
            "an integer resolution from 1 to 1024",
        ));
    }
    Ok(value as u32)
}

fn repeat_param(modifier: &Modifier, name: &str) -> Result<u32> {
    let value = uint_param(modifier, name)?;
    if !(1..=1024).contains(&value) {
        return Err(invalid_parameter(
            modifier,
            name,
            "an integer from 1 to 1024",
        ));
    }
    Ok(value as u32)
}

fn seed_param(modifier: &Modifier, name: &str) -> Result<u32> {
    schema_value(modifier, name)?
        .as_u64()
        .filter(|seed| *seed <= 2_147_483_647)
        .map(|seed| seed as u32)
        .ok_or_else(|| invalid_parameter(modifier, name, "an integer from 0 to 2147483647"))
}

#[cfg(test)]
mod tests {
    use super::{OceanBakeSequence, ocean, sample_cache_uv};
    use crate::{
        eval::ModifierEvaluationContext,
        geom::{GridParams, Mesh},
        model::{Id, Modifier},
    };
    use serde_json::{Map, Value, json};

    fn modifier(params: serde_json::Value) -> Result<Modifier, Box<dyn std::error::Error>> {
        Ok(Modifier {
            id: Id::new("ocean_test")?,
            modifier_type: "ocean".to_owned(),
            name: "Ocean".to_owned(),
            enabled: true,
            params: serde_json::from_value::<Map<String, serde_json::Value>>(params)?,
            binding_data: None,
            baked_ocean_cache: None,
            runtime: crate::model::ModifierRuntime::default(),
            show_viewport: true,
            show_render: true,
        })
    }

    #[test]
    fn generated_grid_has_blender_dimensions_and_corner_uvs()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = Mesh::grid(GridParams {
            size_x: 1.0,
            size_y: 1.0,
            x_subdivisions: 2,
            y_subdivisions: 2,
        })?;
        let out = ocean(
            &source,
            &modifier(
                json!({"geometry_mode":"GENERATE","resolution":3,"viewport_resolution":3,"spatial_size":8,"time":0.5,"random_seed":1}),
            )?,
            1.0,
            ModifierEvaluationContext::default(),
        )?;
        assert_eq!(out.vertices.len(), 10 * 10);
        assert_eq!(out.faces.len(), 9 * 9);
        let uv_store = super::uv_layers::read(&out);
        assert_eq!(uv_store.layers.len(), 1);
        assert_eq!(uv_store.active.as_deref(), Some("UVMap"));
        assert_eq!(uv_store.render.as_deref(), Some("UVMap"));
        assert_eq!(uv_store.layers[0].faces[0].as_ref().unwrap()[0], [0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn live_ocean_uses_evaluated_modifier_time() -> Result<(), Box<dyn std::error::Error>> {
        let source = Mesh::grid(GridParams {
            size_x: 8.0,
            size_y: 8.0,
            x_subdivisions: 9,
            y_subdivisions: 9,
        })?;
        let modifier = modifier(json!({
            "geometry_mode":"DISPLACE",
            "resolution":4,
            "viewport_resolution":4,
            "spatial_size":8,
            "random_seed":12,
            "choppiness":0.0,
            "time":0.5
        }))?;
        let stored_time = ocean(
            &source,
            &modifier,
            1.0,
            ModifierEvaluationContext::default(),
        )?;
        let animated_time = super::ocean_with_parameter_evaluator(
            &source,
            &modifier,
            1.0,
            ModifierEvaluationContext::default(),
            |parameter, base| {
                assert_eq!(parameter, "time");
                Ok(Some(base + 1.5))
            },
        )?;
        assert!(
            stored_time
                .vertices
                .iter()
                .zip(&animated_time.vertices)
                .any(|(stored, animated)| (stored.co.z - animated.co.z).abs() > 1.0e-8)
        );
        Ok(())
    }

    #[test]
    fn spectrum_is_seeded_and_choppiness_changes_horizontal_positions()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = Mesh::grid(GridParams {
            size_x: 8.0,
            size_y: 8.0,
            x_subdivisions: 9,
            y_subdivisions: 9,
        })?;
        let first = ocean(
            &source,
            &modifier(
                json!({"geometry_mode":"DISPLACE","resolution":4,"viewport_resolution":4,"spatial_size":8,"random_seed":12,"choppiness":0.0,"time":2.0}),
            )?,
            1.0,
            ModifierEvaluationContext::default(),
        )?;
        let repeated = ocean(
            &source,
            &modifier(
                json!({"geometry_mode":"DISPLACE","resolution":4,"viewport_resolution":4,"spatial_size":8,"random_seed":12,"choppiness":0.0,"time":2.0}),
            )?,
            1.0,
            ModifierEvaluationContext::default(),
        )?;
        assert_eq!(first, repeated);
        let chopped = ocean(
            &source,
            &modifier(
                json!({"geometry_mode":"DISPLACE","resolution":4,"viewport_resolution":4,"spatial_size":8,"random_seed":12,"choppiness":2.0,"time":2.0}),
            )?,
            1.0,
            ModifierEvaluationContext::default(),
        )?;
        assert!(
            first
                .vertices
                .iter()
                .zip(&chopped.vertices)
                .any(|(a, b)| (a.co.x - b.co.x).abs() + (a.co.y - b.co.y).abs() > 1e-8)
        );
        assert!(
            first
                .vertices
                .iter()
                .any(|vertex| vertex.co.z.abs() > 1e-10)
        );
        Ok(())
    }

    #[test]
    fn foam_and_spray_are_corner_byte_color_layers_with_source_names()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = Mesh::grid(GridParams {
            size_x: 1.0,
            size_y: 1.0,
            x_subdivisions: 2,
            y_subdivisions: 2,
        })?;
        let out = ocean(
            &source,
            &modifier(
                json!({"geometry_mode":"GENERATE","resolution":2,"viewport_resolution":2,"spatial_size":8,"random_seed":12,"time":0.5,"use_foam":true,"use_spray":true,"foam_layer_name":"Foam","spray_layer_name":"Spray","foam_coverage":0.2}),
            )?,
            1.0,
            ModifierEvaluationContext::default(),
        )?;
        let store = super::color_layers::read(&out);
        for name in ["Foam", "Spray"] {
            let layer = store
                .layers
                .iter()
                .find(|layer| layer.name == name)
                .ok_or("Ocean color layer is missing")?;
            assert_eq!(layer.domain, super::color_layers::ColorDomain::Corner);
            assert_eq!(
                out.attributes["blender_metadata"]["color_data_types"][name],
                "BYTE_COLOR"
            );
        }
        Ok(())
    }

    #[test]
    fn ocean_colors_avoid_names_of_canonical_uv_layers() -> Result<(), Box<dyn std::error::Error>> {
        let source = Mesh::grid(GridParams {
            size_x: 1.0,
            size_y: 1.0,
            x_subdivisions: 2,
            y_subdivisions: 2,
        })?;
        let output = ocean(
            &source,
            &modifier(json!({
                "geometry_mode":"GENERATE","resolution":2,"viewport_resolution":2,
                "spatial_size":8,"random_seed":12,"time":0.5,"use_foam":true,
                "foam_layer_name":"UVMap"
            }))?,
            1.0,
            ModifierEvaluationContext::default(),
        )?;
        let colors = super::color_layers::read(&output);
        assert!(colors.layers.iter().any(|layer| layer.name == "UVMap.001"));
        assert!(
            super::uv_layers::read(&output)
                .layers
                .iter()
                .any(|layer| layer.name == "UVMap")
        );
        Ok(())
    }
    #[test]
    fn seeded_phillips_height_and_corner_colors_match_blender_522()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = Mesh::grid(GridParams {
            size_x: 1.0,
            size_y: 1.0,
            x_subdivisions: 2,
            y_subdivisions: 2,
        })?;
        let output = ocean(
            &source,
            &modifier(json!({
                "geometry_mode":"GENERATE","resolution":4,"viewport_resolution":4,
                "spatial_size":50,"size":1.0,"repeat_x":1,"repeat_y":1,
                "random_seed":1,"time":0.5,"wave_scale":1.0,"wave_scale_min":0.01,
                "choppiness":1.0,"wind_velocity":30.0,"damping":0.5,
                "wave_alignment":0.0,"wave_direction":0.0,"depth":200.0,
                "spectrum":"PHILLIPS","fetch_jonswap":120.0,"sharpen_peak_jonswap":0.0,
                "use_foam":true,"use_spray":true,"invert_spray":false,
                "foam_coverage":0.0,"foam_layer_name":"Foam","spray_layer_name":"Spray"
            }))?,
            1.0,
            ModifierEvaluationContext::default(),
        )?;
        let first = output.vertices[0].co;
        assert!((first.x + 25.0).abs() < 1.0e-6);
        assert!((first.y + 25.0).abs() < 1.0e-6);
        assert!((first.z + 0.10027442872524261).abs() < 2.0e-6);
        let colors = super::color_layers::read(&output);
        let foam = colors
            .layers
            .iter()
            .find(|layer| layer.name == "Foam")
            .ok_or("foam layer is missing")?;
        assert!((foam.values[3][0] - 0.010960093699395657).abs() < 1.0e-7);
        let spray = colors
            .layers
            .iter()
            .find(|layer| layer.name == "Spray")
            .ok_or("spray layer is missing")?;
        assert!((spray.values[0][0] - 0.838799238204956).abs() < 1.0e-6);
        assert!((spray.values[0][2] - 0.1119324341416359).abs() < 1.0e-6);
        Ok(())
    }
    // These are Blender's baked EXR pixels, not evaluated cached meshes: background baking returns
    // before the job end callback sets `is_cached`.
    #[test]
    fn baked_sequence_matches_blender_522_exr_pixels() -> Result<(), Box<dyn std::error::Error>> {
        let corpus: Value = serde_json::from_str(include_str!(
            "../../../../../tests/fixtures/ocean_parity/corpus.json"
        ))?;
        for case in corpus["cases"]
            .as_array()
            .ok_or("Ocean corpus cases are missing")?
        {
            let Some(files) = case["exr_files"].as_object() else {
                continue;
            };
            let modifier = modifier(case["params"].clone())?;
            let mut sequence = OceanBakeSequence::new(&modifier)?;
            let start = case["params"]["frame_start"]
                .as_i64()
                .ok_or("bake frame_start is missing")? as i32;
            let end = case["params"]["frame_end"]
                .as_i64()
                .ok_or("bake frame_end is missing")? as i32;
            let base_time = case["params"]["time"]
                .as_f64()
                .ok_or("bake time is missing")? as f32;
            let frame_times =
                case["observations"]["animated_time"]["modifier_time_by_frame"].as_object();
            if let Some(animated_time) = case["observations"]["animated_time"].as_object() {
                let data_path = animated_time["time_data_path"]
                    .as_str()
                    .ok_or("animated Ocean time path is missing")?;
                assert!(
                    animated_time["channel_curves"]
                        .as_array()
                        .is_some_and(|curves| curves.iter().any(|curve| {
                            curve["data_path"].as_str() == Some(data_path)
                                && curve["array_index"].as_u64() == Some(0)
                        })),
                    "{} has no ActionSlot F-Curve for its actual modifier time path",
                    case["name"]
                );
                assert!(
                    animated_time["action_slot_identifier"]
                        .as_str()
                        .is_some_and(|id| !id.is_empty()),
                    "{} animated Ocean Action has no slot identity",
                    case["name"]
                );
                let times = frame_times.ok_or("animated Ocean frame-time map is missing")?;
                assert!(
                    times.len() > 1,
                    "{} has no per-frame animated time samples",
                    case["name"]
                );
                let start_time = times
                    .get(&start.to_string())
                    .and_then(Value::as_f64)
                    .ok_or("animated Ocean first-frame time is missing")?;
                let end_time = times
                    .get(&end.to_string())
                    .and_then(Value::as_f64)
                    .ok_or("animated Ocean last-frame time is missing")?;
                assert_ne!(
                    start_time, end_time,
                    "{} animated Ocean time did not change over its bake frames",
                    case["name"]
                );
            }
            for frame in start..=end {
                let time = match frame_times {
                    Some(times) => times
                        .get(&frame.to_string())
                        .and_then(Value::as_f64)
                        .ok_or("animated Ocean frame time is missing")?
                        as f32,
                    None => base_time,
                };
                let actual = sequence.bake_frame(frame, time)?;
                let suffix = format!("_{frame:04}.exr");
                let expected_count = files.keys().filter(|name| name.ends_with(&suffix)).count();
                assert_eq!(
                    actual.len(),
                    expected_count,
                    "{} frame {frame} channel count",
                    case["name"]
                );
                for image in actual {
                    let expected = &files[&image.filename];
                    assert_eq!(
                        expected["width"].as_u64(),
                        Some(image.resolution as u64),
                        "{} {} width",
                        case["name"],
                        image.filename
                    );
                    assert_eq!(
                        expected["height"].as_u64(),
                        Some(image.resolution as u64),
                        "{} {} height",
                        case["name"],
                        image.filename
                    );
                    assert!(
                        expected["stored_channel_types"]
                            .as_object()
                            .is_some_and(|types| types.values().all(|value| value == "half")),
                        "{} {} is not Blender half-float EXR",
                        case["name"],
                        image.filename
                    );
                    let expected_pixels = expected["pixel_values_rgba_row_major"]
                        .as_array()
                        .ok_or("baked EXR pixels are missing")?;
                    assert_eq!(
                        expected_pixels.len(),
                        image.pixels.len() * 4,
                        "{} {} pixel count",
                        case["name"],
                        image.filename
                    );
                    for (pixel, actual) in image.pixels.iter().enumerate() {
                        for (channel, actual) in actual.iter().enumerate() {
                            let expected = expected_pixels[pixel * 4 + channel]
                                .as_f64()
                                .ok_or("baked EXR channel is not numeric")?
                                as f32;
                            let rounded = exr::prelude::f16::from_f32(*actual).to_f32();
                            let half = exr::prelude::f16::from_f32(*actual);
                            let bits = half.to_bits();
                            let ulp = if bits & 0x7fff == 0 {
                                exr::prelude::f16::from_bits(1).to_f32()
                            } else if bits & 0x8000 == 0 {
                                (exr::prelude::f16::from_bits(bits + 1).to_f32() - rounded).abs()
                            } else {
                                (exr::prelude::f16::from_bits(bits - 1).to_f32() - rounded).abs()
                            };
                            assert!(
                                (expected - rounded).abs() <= ulp,
                                "{} {} pixel {pixel} channel {channel}: native {actual} -> {rounded}, Blender {expected}, half ULP {ulp}",
                                case["name"],
                                image.filename,
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn cached_ocean_sampling_wraps_uv_and_extends_image_edges() {
        let pixels = [
            [0.0, 10.0, 20.0, 1.0],
            [2.0, 12.0, 22.0, 1.0],
            [4.0, 14.0, 24.0, 1.0],
            [6.0, 16.0, 26.0, 1.0],
        ];
        assert_eq!(
            sample_cache_uv(&pixels, 2, 2, 0.5, 0.5, 2, 2),
            [3.0, 13.0, 23.0, 1.0]
        );
        assert_eq!(sample_cache_uv(&pixels, 2, 2, 0.0, 0.0, 2, 2), pixels[0]);
        assert_eq!(
            sample_cache_uv(&pixels, 2, 2, 0.875, 0.5, 2, 2),
            [4.0, 14.0, 24.0, 1.0]
        );
        assert_eq!(
            sample_cache_uv(&pixels, 2, 2, -0.125, 0.5, 2, 2),
            [4.0, 14.0, 24.0, 1.0]
        );
    }
}
