//! The world's radiance captured once per render.
//!
//! The raw cube remains available to Path. EEVEE also builds its source five-level octahedral sphere-probe atlas
//! from that stored cube, matching Blender's bilinear atlas sampling, GGX convolution, roughness-to-LOD mapping,
//! and mip mix. Its world graph is evaluated only during the capture, not for every surface lookup.

use std::{collections::BTreeMap, sync::OnceLock};

use glam::{DVec3, Vec3, Vec4};

use super::eevee_fast_gi::FastGiSh;

use crate::{
    error::{ErrorCode, PotError, Result},
    graph::NodeGroup,
    image::ImageData,
    model::{Registry, World},
    shader::{EvaluatorWorkspace, evaluate_world_with_workspace},
};

/// Cycles/Path's captured cube remains at its existing face extent.
const FACE_TEXELS: usize = 64;
const SPHERE_PROBE_REMAP_GROUP_SIZE: usize = 32;
const SPHERE_PROBE_SH_GROUP_SIZE: usize = 256;
/// Values mirror Blender 5.2.2 `eevee_defines.hh` and `eevee_lightprobe_sphere.bsl.hh`.
const EEVEE_PROBE_MIPMAP_LEVELS: usize = 5;
const EEVEE_PROBE_MIP_MAX_ROUGHNESS: f32 = 0.7;
const EEVEE_PROBE_MIP_SAMPLE_COUNT: u32 = 196;

/// Path keeps its existing raw cube; EEVEE uses the scene's source cubemap resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum WorldProbeMode {
    Path,
    Eevee {
        cubemap_resolution: u32,
        surface_indirect_clamp: f32,
    },
}

/// `SphereProbeModule::probe_render_extent()` in Blender 5.2.2.
fn eevee_probe_face_texels(cubemap_resolution: u32) -> usize {
    (cubemap_resolution / 2) as usize
}
fn eevee_probe_sun_threshold(world: &World) -> Result<f32> {
    let threshold = world.eevee_sun_threshold.ok_or_else(|| {
        PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "EEVEE world probe lighting needs the source World.sun_threshold; reimport the Blender scene to recover it",
            serde_json::json!({"feature_id":"world.eevee.sun_threshold"}),
        )
    })? as f32;
    Ok(if threshold == 0.0 { 1.0e20 } else { threshold })
}

impl WorldProbeMode {
    fn face_texels(self) -> usize {
        match self {
            Self::Path => FACE_TEXELS,
            Self::Eevee {
                cubemap_resolution, ..
            } => eevee_probe_face_texels(cubemap_resolution),
        }
    }
}

type EeveeProbeTexel = [exr::prelude::f16; 4];

enum WorldProbeRadiance {
    Path(Vec<DVec3>),
    Eevee {
        cubemap: Vec<EeveeProbeTexel>,
        sun_threshold: f32,
        surface_indirect_clamp: f32,
    },
}
/// The GPU-extracted EEVEE world Sun radiance and safe-normalized source direction.
/// CPU World settings separately supply the UI-derived surface shape and volume-power placeholder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct EeveeWorldSun {
    pub radiance: Vec3,
    pub direction: Vec3,
}
/// The world's radiance by direction, sampled at the centre of each cube-map texel.
pub(super) struct WorldProbe {
    radiance: WorldProbeRadiance,
    face_texels: usize,
    /// EEVEE's octahedral reflection-probe atlas, stored as source `SFLOAT16x4` texels.
    eevee_prefiltered: Option<[Vec<EeveeProbeTexel>; EEVEE_PROBE_MIPMAP_LEVELS]>,
    /// EEVEE's non-directional spherical-harmonic L0 for sphere-thickness and isotropic volume lighting.
    average_irradiance: DVec3,
    /// Source sphere-probe excess radiance reduced to its world Sun.
    world_sun: Option<EeveeWorldSun>,
    /// Source-basis SH, projected on demand from the captured EEVEE SFLOAT16 world cube.
    fast_gi_sh: OnceLock<FastGiSh>,
}

impl WorldProbe {
    /// Evaluates the world at every texel centre.
    pub(super) fn build(
        world: &World,
        groups: &Registry<NodeGroup>,
        images: &BTreeMap<String, ImageData>,
        mode: WorldProbeMode,
    ) -> Result<Self> {
        let face_texels = mode.face_texels();
        let (radiance, eevee_prefiltered, average_irradiance, world_sun) = match mode {
            WorldProbeMode::Path => {
                let mut radiance = Vec::with_capacity(6 * face_texels * face_texels);
                let average_irradiance =
                    sample_world_probe(world, groups, images, face_texels, |value| {
                        radiance.push(value);
                        value
                    })?;
                (
                    WorldProbeRadiance::Path(radiance),
                    None,
                    average_irradiance,
                    None,
                )
            }
            WorldProbeMode::Eevee {
                surface_indirect_clamp,
                ..
            } => {
                let sun_threshold = eevee_probe_sun_threshold(world)?;
                let surface_indirect_clamp =
                    eevee_probe_clamp_surface_indirect(surface_indirect_clamp);
                let mut radiance = Vec::with_capacity(6 * face_texels * face_texels);
                let average_irradiance =
                    sample_world_probe(world, groups, images, face_texels, |value| {
                        let texel = store_eevee_probe_texel(value);
                        radiance.push(texel);
                        brightness_clamp_max(load_eevee_probe_texel(texel), sun_threshold)
                            .as_dvec3()
                    })?;
                let extract_sun = world
                    .eevee_sun_threshold
                    .is_some_and(|threshold| threshold as f32 > 0.0);
                let (eevee_prefiltered, world_sun) = build_eevee_prefiltered_atlas(
                    &radiance,
                    face_texels,
                    sun_threshold,
                    surface_indirect_clamp,
                    extract_sun,
                );
                (
                    WorldProbeRadiance::Eevee {
                        cubemap: radiance,
                        surface_indirect_clamp,
                        sun_threshold,
                    },
                    Some(eevee_prefiltered),
                    average_irradiance,
                    world_sun,
                )
            }
        };
        Ok(Self {
            radiance,
            face_texels,
            eevee_prefiltered,
            average_irradiance,
            world_sun,
            fast_gi_sh: OnceLock::new(),
        })
    }
    /// Samples the captured cube without spatial filtering; EEVEE applies its source sun threshold.
    pub(super) fn radiance(&self, direction: DVec3) -> DVec3 {
        let (face, s, t) = cube_coordinates(direction);
        let column = texel_at(s, self.face_texels);
        let row = texel_at(t, self.face_texels);
        match &self.radiance {
            WorldProbeRadiance::Path(cubemap) => {
                cubemap[(face * self.face_texels + row) * self.face_texels + column]
            }
            WorldProbeRadiance::Eevee {
                cubemap,
                sun_threshold,
                surface_indirect_clamp,
            } => {
                let radiance = load_eevee_probe_texel(
                    cubemap[(face * self.face_texels + row) * self.face_texels + column],
                );
                brightness_clamp_max(
                    brightness_clamp_max(radiance, *sun_threshold),
                    *surface_indirect_clamp,
                )
                .as_dvec3()
            }
        }
    }

    /// EEVEE's bilinear octahedral-atlas sample with source roughness-to-LOD mapping and mip mix.
    /// Returns `None` when this probe was built without EEVEE prefiltering.
    pub(super) fn prefiltered_radiance(&self, direction: DVec3, roughness: f64) -> Option<DVec3> {
        let mips = self.eevee_prefiltered.as_ref()?;
        let lod = eevee_roughness_to_lod(roughness as f32);
        let low = lod.floor() as usize;
        let high = lod.ceil() as usize;
        let blend = lod - low as f32;
        let low_sample = octahedral_atlas_sample(
            &mips[low],
            eevee_probe_mip_extent(self.face_texels, low),
            direction,
        );
        let high_sample = octahedral_atlas_sample(
            &mips[high],
            eevee_probe_mip_extent(self.face_texels, high),
            direction,
        );
        Some((low_sample * (1.0 - blend) + high_sample * blend).as_dvec3())
    }

    /// Path's captured average stays raw; EEVEE applies its source indirect clamp after sampling.
    pub(super) fn average_irradiance(&self) -> DVec3 {
        match &self.radiance {
            WorldProbeRadiance::Path(_) => self.average_irradiance,
            WorldProbeRadiance::Eevee {
                surface_indirect_clamp,
                ..
            } => brightness_clamp_max(
                Vec3::new(
                    self.average_irradiance.x as f32,
                    self.average_irradiance.y as f32,
                    self.average_irradiance.z as f32,
                ),
                *surface_indirect_clamp,
            )
            .as_dvec3(),
        }
    }
    /// EEVEE's source-extracted world Sun, absent when the threshold produces no excess light.
    pub(super) fn world_sun(&self) -> Option<EeveeWorldSun> {
        self.world_sun
    }
    /// Returns cached source-order SH when requested; Path probes have no EEVEE coefficients.
    pub(super) fn fast_gi_sh(&self) -> Option<FastGiSh> {
        let WorldProbeRadiance::Eevee {
            cubemap,
            sun_threshold,
            ..
        } = &self.radiance
        else {
            return None;
        };
        Some(
            *self
                .fast_gi_sh
                .get_or_init(|| extract_eevee_world_sh(cubemap, self.face_texels, *sun_threshold)),
        )
    }
}

fn eevee_probe_mip_extent(face_texels: usize, mip: usize) -> usize {
    (2 * face_texels) >> mip
}

fn store_eevee_probe_texel(value: DVec3) -> EeveeProbeTexel {
    [
        exr::prelude::f16::from_f32(value.x as f32),
        exr::prelude::f16::from_f32(value.y as f32),
        exr::prelude::f16::from_f32(value.z as f32),
        exr::prelude::f16::from_f32(1.0),
    ]
}

fn load_eevee_probe_texel(texel: EeveeProbeTexel) -> Vec3 {
    Vec3::new(texel[0].to_f32(), texel[1].to_f32(), texel[2].to_f32())
}
fn brightness_clamp_max(radiance: Vec3, threshold: f32) -> Vec3 {
    let brightness = radiance.max_element();
    if brightness > threshold {
        radiance * (threshold / brightness)
    } else {
        radiance
    }
}
fn eevee_probe_clamp_surface_indirect(value: f32) -> f32 {
    if value > 0.0 { value } else { 1.0e20 }
}

fn build_eevee_prefiltered_atlas(
    cubemap: &[EeveeProbeTexel],
    face_texels: usize,
    sun_threshold: f32,
    surface_indirect_clamp: f32,
    extract_sun: bool,
) -> (
    [Vec<EeveeProbeTexel>; EEVEE_PROBE_MIPMAP_LEVELS],
    Option<EeveeWorldSun>,
) {
    let mut mips = std::array::from_fn(|mip| {
        let extent = eevee_probe_mip_extent(face_texels, mip);
        vec![store_eevee_probe_texel(DVec3::ZERO); extent * extent]
    });
    let extent = eevee_probe_mip_extent(face_texels, 0);
    let groups_per_axis = extent.div_ceil(SPHERE_PROBE_REMAP_GROUP_SIZE);
    let group_count = groups_per_axis * groups_per_axis;
    debug_assert!(group_count <= SPHERE_PROBE_MAX_HARMONIC);
    let mut thread_radiance = [Vec3::ZERO; SPHERE_PROBE_SH_GROUP_SIZE];
    let mut thread_direction = [Vec4::ZERO; SPHERE_PROBE_SH_GROUP_SIZE];
    for group_y in 0..groups_per_axis {
        for group_x in 0..groups_per_axis {
            let mut local_radiance = [Vec3::ZERO; SPHERE_PROBE_REMAP_TEXELS_PER_GROUP];
            let mut local_direction = [Vec4::ZERO; SPHERE_PROBE_REMAP_TEXELS_PER_GROUP];
            for local_index in 0..SPHERE_PROBE_REMAP_TEXELS_PER_GROUP {
                let column = group_x * SPHERE_PROBE_REMAP_GROUP_SIZE
                    + local_index % SPHERE_PROBE_REMAP_GROUP_SIZE;
                let row = group_y * SPHERE_PROBE_REMAP_GROUP_SIZE
                    + local_index / SPHERE_PROBE_REMAP_GROUP_SIZE;
                if column >= extent || row >= extent {
                    continue;
                }
                let direction = octahedral_direction(
                    column as f32 / (extent - 2) as f32,
                    row as f32 / (extent - 2) as f32,
                );
                let radiance = sample_world_cubemap(cubemap, face_texels, direction);
                let thresholded_radiance = brightness_clamp_max(radiance, sun_threshold);
                let stored_radiance =
                    brightness_clamp_max(thresholded_radiance, surface_indirect_clamp);
                mips[0][row * extent + column] =
                    store_eevee_probe_texel(stored_radiance.as_dvec3());

                if extract_sun {
                    let weight = octahedral_texel_solid_angle(column, row, extent);
                    let excess = radiance - thresholded_radiance;
                    let excess_length = excess.length();
                    local_radiance[local_index] = excess * weight;
                    local_direction[local_index] =
                        direction.normalize().extend(1.0) * weight * excess_length;
                }
            }
            if extract_sun {
                let mut stride = SPHERE_PROBE_REMAP_TEXELS_PER_GROUP / 2;
                while stride != 0 {
                    for index in 0..stride {
                        local_radiance[index] += local_radiance[index + stride];
                        local_direction[index] += local_direction[index + stride];
                    }
                    stride /= 2;
                }
                let group = group_y * groups_per_axis + group_x;
                let thread = group % SPHERE_PROBE_SH_GROUP_SIZE;
                thread_radiance[thread] += local_radiance[0];
                thread_direction[thread] += local_direction[0];
            }
        }
    }

    let world_sun = if extract_sun {
        let mut stride = SPHERE_PROBE_SH_GROUP_SIZE / 2;
        while stride != 0 {
            for index in 0..stride {
                thread_radiance[index] += thread_radiance[index + stride];
                thread_direction[index] += thread_direction[index + stride];
            }
            stride /= 2;
        }
        eevee_world_sun_from_reduction(thread_radiance[0], thread_direction[0])
    } else {
        None
    };

    for mip in 1..EEVEE_PROBE_MIPMAP_LEVELS {
        let read_extent = eevee_probe_mip_extent(face_texels, mip - 1);
        let write_extent = eevee_probe_mip_extent(face_texels, mip);
        let (previous_mips, current_mip_and_later) = mips.split_at_mut(mip);
        convolve_eevee_probe_mip(
            &previous_mips[mip - 1],
            read_extent,
            &mut current_mip_and_later[0],
            write_extent,
            mip - 1,
        );
    }
    (mips, world_sun)
}
fn eevee_world_sun_from_reduction(radiance: Vec3, direction_sum: Vec4) -> Option<EeveeWorldSun> {
    if radiance.max_element() <= 0.0 {
        return None;
    }
    let direction_input = direction_sum.truncate() / direction_sum.w;
    let length_squared = direction_input.length_squared();
    let direction = if length_squared > 1.0e-35 {
        direction_input / length_squared.sqrt()
    } else {
        Vec3::X
    };
    Some(EeveeWorldSun {
        radiance,
        direction,
    })
}

fn sample_world_cubemap(cubemap: &[EeveeProbeTexel], face_texels: usize, direction: Vec3) -> Vec3 {
    let (face, s, t) = cube_coordinates(direction.as_dvec3());
    let x = (f64::midpoint(s, 1.0) * face_texels as f64 - 0.5) as f32;
    let y = (f64::midpoint(t, 1.0) * face_texels as f64 - 0.5) as f32;
    let x_floor = x.floor();
    let y_floor = y.floor();
    let x0 = x_floor as i32;
    let y0 = y_floor as i32;
    let fraction_x = x - x_floor;
    let fraction_y = y - y_floor;
    let p00 = world_cubemap_tap(cubemap, face_texels, face, x0, y0);
    let p10 = world_cubemap_tap(cubemap, face_texels, face, x0 + 1, y0);
    let p01 = world_cubemap_tap(cubemap, face_texels, face, x0, y0 + 1);
    let p11 = world_cubemap_tap(cubemap, face_texels, face, x0 + 1, y0 + 1);
    let lower = p00 * (1.0 - fraction_x) + p10 * fraction_x;
    let upper = p01 * (1.0 - fraction_x) + p11 * fraction_x;
    lower * (1.0 - fraction_y) + upper * fraction_y
}

fn world_cubemap_tap(
    cubemap: &[EeveeProbeTexel],
    face_texels: usize,
    face: usize,
    column: i32,
    row: i32,
) -> Vec3 {
    let extent = face_texels as i32;
    if (0..extent).contains(&column) && (0..extent).contains(&row) {
        return load_eevee_probe_texel(
            cubemap[(face * face_texels + row as usize) * face_texels + column as usize],
        );
    }
    let s = 2.0 * (column as f64 + 0.5) / f64::from(extent) - 1.0;
    let t = 2.0 * (row as f64 + 0.5) / f64::from(extent) - 1.0;
    let direction = face_direction(face, s, t);
    let (face, s, t) = cube_coordinates(direction);
    let column = texel_at(s, face_texels);
    let row = texel_at(t, face_texels);
    load_eevee_probe_texel(cubemap[(face * face_texels + row) * face_texels + column])
}

/// Mirrors Blender 5.2.2 `eevee_lightprobe_sphere_bake.bsl.hh::mip_convolve`.
fn convolve_eevee_probe_mip(
    input: &[EeveeProbeTexel],
    input_extent: usize,
    output: &mut [EeveeProbeTexel],
    output_extent: usize,
    read_lod: usize,
) {
    let previous_roughness = eevee_lod_to_roughness(read_lod as f32);
    let current_roughness = eevee_lod_to_roughness((read_lod + 1) as f32);
    let mip_roughness = (current_roughness.powi(3) - previous_roughness.powi(3))
        .cbrt()
        .max(0.02);
    let cone_cos = eevee_convolution_cone_cosine(mip_roughness);
    let output_denominator = (output_extent - 2) as f32;
    for row in 0..output_extent {
        for column in 0..output_extent {
            let direction = octahedral_direction(
                column as f32 / output_denominator,
                row as f32 / output_denominator,
            )
            .normalize();
            let (tangent, bitangent) = eevee_tangent_basis(direction);
            let mut weight_sum = 0.0_f32;
            let mut radiance_sum = Vec3::ZERO;
            for sample in 0..EEVEE_PROBE_MIP_SAMPLE_COUNT {
                let rand = hammersley_2d(sample, EEVEE_PROBE_MIP_SAMPLE_COUNT);
                let sample_direction = sample_uniform_cone(rand, cone_cos);
                let incoming = tangent * sample_direction.x
                    + bitangent * sample_direction.y
                    + direction * sample_direction.z;
                let radiance = octahedral_atlas_sample(input, input_extent, incoming.as_dvec3());
                let weight = eevee_convolution_sample_weight(direction, incoming, mip_roughness);
                radiance_sum += radiance * weight;
                weight_sum += weight;
            }
            output[row * output_extent + column] =
                store_eevee_probe_texel((radiance_sum / weight_sum).as_dvec3());
        }
    }
}

fn hammersley_2d(index: u32, sample_count: u32) -> (f32, f32) {
    (
        index as f32 / sample_count as f32,
        index.reverse_bits() as f32 * 2.328_306_4e-10,
    )
}

fn sample_uniform_cone(rand: (f32, f32), cone_cos: f32) -> Vec3 {
    let cos_theta = cone_cos + (1.0 - cone_cos) * rand.0;
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = (rand.1 - 0.5) * std::f32::consts::TAU;
    let cos_phi = phi.cos();
    let sin_phi = (1.0 - cos_phi * cos_phi).max(0.0).sqrt()
        * if phi > 0.0 {
            1.0
        } else if phi < 0.0 {
            -1.0
        } else {
            0.0
        };
    Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)
}

fn eevee_tangent_basis(up: Vec3) -> (Vec3, Vec3) {
    let z_sign = if up.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (z_sign + up.z);
    let b = up.x * up.y * a;
    (
        Vec3::new(1.0 + z_sign * up.x * up.x * a, z_sign * b, -z_sign * up.x),
        Vec3::new(b, z_sign + up.y * up.y * a, -up.y),
    )
}

fn eevee_convolution_sample_weight(out_direction: Vec3, in_direction: Vec3, roughness: f32) -> f32 {
    let normal = out_direction.normalize();
    let incoming = in_direction.normalize();
    let half_vector = (normal + incoming).normalize();
    let nh = normal.dot(half_vector).clamp(0.0, 1.0);
    let m = roughness * roughness;
    (2.0 * (nh - 1.0) / (m * m)).exp()
}

fn eevee_convolution_cone_cosine(roughness: f32) -> f32 {
    let m = roughness * roughness;
    let cutoff = 0.01 + (0.14 - 0.01) * m;
    let half_angle_cos = 1.0 + cutoff.ln() * m * m * 0.5;
    let half_angle_sin_squared = (1.0 - half_angle_cos * half_angle_cos).max(0.0);
    half_angle_cos * half_angle_cos - half_angle_sin_squared
}

fn eevee_lod_to_roughness(lod: f32) -> f32 {
    let mip_ratio = lod / (EEVEE_PROBE_MIPMAP_LEVELS - 1) as f32;
    let b = 0.6_f32;
    let c = 0.4_f32;
    let b2 = b * b;
    let c2 = c * c;
    let ratio =
        (-(4.0 * mip_ratio * b * c2 + c2 * c2).sqrt() + 2.0 * mip_ratio * b + c2) / (2.0 * b2);
    ratio * EEVEE_PROBE_MIP_MAX_ROUGHNESS
}

fn eevee_roughness_to_lod(roughness: f32) -> f32 {
    let ratio = (roughness / EEVEE_PROBE_MIP_MAX_ROUGHNESS).clamp(0.0, 1.0);
    let mip_ratio = ratio + (ratio.sqrt() - ratio) * 0.4;
    mip_ratio * (EEVEE_PROBE_MIPMAP_LEVELS - 1) as f32
}

fn octahedral_uv_from_direction(direction: Vec3) -> (f32, f32) {
    let mut x = direction.x;
    let mut y = direction.y;
    let z = direction.z;
    let scale = x.abs() + y.abs() + z.abs();
    x /= scale;
    y /= scale;
    let z = z / scale;
    if z < 0.0 {
        let sign_x = if x >= 0.0 { 1.0 } else { -1.0 };
        let sign_y = if y >= 0.0 { 1.0 } else { -1.0 };
        let old_x = x;
        x = (1.0 - y.abs()) * sign_x;
        y = (1.0 - old_x.abs()) * sign_y;
    }
    (x * 0.5 + 0.5, y * 0.5 + 0.5)
}

fn octahedral_atlas_sample(atlas: &[EeveeProbeTexel], extent: usize, direction: DVec3) -> Vec3 {
    let (u, v) = octahedral_uv_from_direction(Vec3::new(
        direction.x as f32,
        direction.y as f32,
        direction.z as f32,
    ));
    let scale = (extent - 2) as f32 / extent as f32;
    let offset = 0.5 / extent as f32;
    let x = ((u * scale + offset) * extent as f32 - 0.5).clamp(0.0, (extent - 2) as f32);
    let y = ((v * scale + offset) * extent as f32 - 0.5).clamp(0.0, (extent - 2) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(extent - 2);
    let y1 = (y0 + 1).min(extent - 2);
    let fraction_x = x - x.floor();
    let fraction_y = y - y.floor();
    let tap = |column: usize, row: usize| load_eevee_probe_texel(atlas[row * extent + column]);
    let lower = tap(x0, y0) * (1.0 - fraction_x) + tap(x1, y0) * fraction_x;
    let upper = tap(x0, y1) * (1.0 - fraction_x) + tap(x1, y1) * fraction_x;
    lower * (1.0 - fraction_y) + upper * fraction_y
}

fn texel_at(coordinate: f64, extent: usize) -> usize {
    let index = (f64::midpoint(coordinate, 1.0) * extent as f64).floor();
    (index.max(0.0) as usize).min(extent - 1)
}

/// EEVEE's isotropic volume-world result: project the half-float sphere probe to L0, then dot
/// its half-float volume-probe coefficient with the isotropic phase (`eevee_volume.bsl.hh`).
/// The second result reports sampled directional variation that can contribute through SH L1.
/// `cubemap_resolution` is EEVEE's scene setting; Blender's probe producer captures at half this extent.
pub(super) fn volume_ambient_l0(
    world: &World,
    groups: &Registry<NodeGroup>,
    images: &BTreeMap<String, ImageData>,
    cubemap_resolution: u32,
) -> Result<(DVec3, bool)> {
    let mut workspace = EvaluatorWorkspace::default();
    let face_texels = eevee_probe_face_texels(cubemap_resolution);
    let mut cubemap: Vec<EeveeProbeTexel> = Vec::with_capacity(6 * face_texels * face_texels);
    let mut first_radiance: Option<[f32; 3]> = None;
    let mut nonuniform = false;
    for face in 0..6 {
        for row in 0..face_texels {
            for column in 0..face_texels {
                let direction = face_direction(
                    face,
                    texel_centre_at(column, face_texels),
                    texel_centre_at(row, face_texels),
                );
                let radiance = evaluate_world_with_workspace(
                    world,
                    groups,
                    images,
                    direction,
                    false,
                    &mut workspace,
                )?;
                let radiance = store_eevee_probe_texel(radiance);
                let radiance_rgb = [
                    radiance[0].to_f32(),
                    radiance[1].to_f32(),
                    radiance[2].to_f32(),
                ];
                if let Some(first) = first_radiance {
                    nonuniform |= first.map(f32::to_bits) != radiance_rgb.map(f32::to_bits);
                } else {
                    first_radiance = Some(radiance_rgb);
                }
                cubemap.push(radiance);
            }
        }
    }

    // EEVEE projects the half-float cubemap through its double-resolution octahedral atlas.
    let extent = 2 * face_texels;
    let groups_per_axis = extent / SPHERE_PROBE_REMAP_GROUP_SIZE;
    let mut coefficients = vec![[0.0_f32; 3]; groups_per_axis * groups_per_axis];
    let mut samples = [[0.0_f32; 3]; SPHERE_PROBE_REMAP_GROUP_SIZE * SPHERE_PROBE_REMAP_GROUP_SIZE];
    let l0 = 0.282_094_8_f32;
    for group_y in 0..groups_per_axis {
        for group_x in 0..groups_per_axis {
            for local_y in 0..SPHERE_PROBE_REMAP_GROUP_SIZE {
                for local_x in 0..SPHERE_PROBE_REMAP_GROUP_SIZE {
                    let local_index = local_y * SPHERE_PROBE_REMAP_GROUP_SIZE + local_x;
                    let column = group_x * SPHERE_PROBE_REMAP_GROUP_SIZE + local_x;
                    let row = group_y * SPHERE_PROBE_REMAP_GROUP_SIZE + local_y;
                    let weight = octahedral_texel_solid_angle(column, row, extent);
                    if weight == 0.0 {
                        samples[local_index] = [0.0; 3];
                        continue;
                    }
                    let denominator = (extent - 2) as f32;
                    let direction =
                        octahedral_direction(column as f32 / denominator, row as f32 / denominator);
                    let radiance = cube_probe_sample(&cubemap, face_texels, direction);
                    samples[local_index] = radiance.map(|channel| channel * weight);
                }
            }
            let mut stride = samples.len() / 2;
            while stride != 0 {
                for index in 0..stride {
                    let source = samples[index + stride];
                    for (target, source) in samples[index].iter_mut().zip(source) {
                        *target += source;
                    }
                }
                stride /= 2;
            }
            coefficients[group_y * groups_per_axis + group_x] =
                samples[0].map(|radiance| radiance * l0);
        }
    }
    let coefficient = sum_sphere_probe_coefficients(&coefficients);
    Ok((
        DVec3::new(
            f64::from(exr::prelude::f16::from_f32(coefficient[0]).to_f32() * l0),
            f64::from(exr::prelude::f16::from_f32(coefficient[1]).to_f32() * l0),
            f64::from(exr::prelude::f16::from_f32(coefficient[2]).to_f32() * l0),
        ),
        nonuniform,
    ))
}

fn sum_sphere_probe_coefficients(coefficients: &[[f32; 3]]) -> [f32; 3] {
    let mut sums_by_thread = [[0.0_f32; 3]; SPHERE_PROBE_SH_GROUP_SIZE];
    let batches = coefficients.len().div_ceil(SPHERE_PROBE_SH_GROUP_SIZE);
    for batch in 0..batches {
        for (local_index, thread_sum) in sums_by_thread.iter_mut().enumerate() {
            let index = batch * SPHERE_PROBE_SH_GROUP_SIZE + local_index;
            if let Some(coefficient) = coefficients.get(index) {
                for (target, source) in thread_sum.iter_mut().zip(coefficient) {
                    *target += *source;
                }
            }
        }
    }
    let mut stride = SPHERE_PROBE_SH_GROUP_SIZE / 2;
    while stride != 0 {
        for index in 0..stride {
            let source = sums_by_thread[index + stride];
            for (target, source) in sums_by_thread[index].iter_mut().zip(source) {
                *target += source;
            }
        }
        stride /= 2;
    }
    sums_by_thread[0]
}
fn octahedral_direction(u: f32, v: f32) -> Vec3 {
    let x = 2.0 * u - 1.0;
    let y = 2.0 * v - 1.0;
    let mut direction = Vec3::new(x, y, 1.0 - (x.abs() + y.abs()));
    if x.abs() + y.abs() > 1.0 {
        let sign_x = if x >= 0.0 { 1.0 } else { -1.0 };
        let sign_y = if y >= 0.0 { 1.0 } else { -1.0 };
        direction.x = (y.abs() - 1.0) * -sign_x;
        direction.y = (x.abs() - 1.0) * -sign_y;
    }
    direction
}

fn octahedral_texel_solid_angle(column: usize, row: usize, extent: usize) -> f32 {
    if column == extent - 1 || row == extent - 1 {
        return 0.0;
    }
    let half = extent / 2;
    let padded = extent - 2;
    let wrapped_x = if column >= half {
        padded - column
    } else {
        column
    };
    let wrapped_y = if row >= half { padded - row } else { row };
    let clamp = |coordinate: f32| coordinate.clamp(0.0, (half - 1) as f32);
    let direction = |x: f32, y: f32| {
        octahedral_direction(
            clamp(x) / (extent - 2) as f32,
            clamp(y) / (extent - 2) as f32,
        )
        .normalize()
    };
    let x = wrapped_x as f32;
    let y = wrapped_y as f32;
    let v00 = direction(x - 0.5, y - 0.5);
    let v10 = direction(x + 0.5, y - 0.5);
    let v01 = direction(x - 0.5, y + 0.5);
    let v11 = direction(x + 0.5, y + 0.5);
    let triangle_area = |a: Vec3, b: Vec3, c: Vec3| {
        2.0 * a
            .dot(b.cross(c))
            .abs()
            .atan2(1.0 + b.dot(c) + a.dot(c) + a.dot(b))
    };
    let mut area = triangle_area(v00, v10, v01) + triangle_area(v01, v10, v11);
    if wrapped_x == half - 1 {
        area *= 2.0;
    }
    if wrapped_y == half - 1 {
        area *= 2.0;
    }
    area
}

const SPHERE_PROBE_MAX_HARMONIC: usize = (1 << 12) * (1 << 12) / 32 / 32;
const SPHERE_PROBE_REMAP_TEXELS_PER_GROUP: usize =
    SPHERE_PROBE_REMAP_GROUP_SIZE * SPHERE_PROBE_REMAP_GROUP_SIZE;

/// Mirrors the sphere-probe remap SH path over captured SFLOAT16 radiance and source clamp.
fn extract_eevee_world_sh(
    cubemap: &[EeveeProbeTexel],
    face_texels: usize,
    sun_threshold: f32,
) -> FastGiSh {
    let extent = 2 * face_texels;
    let groups_per_axis = extent / SPHERE_PROBE_REMAP_GROUP_SIZE;
    let mut sums_by_thread = [FastGiSh::ZERO; SPHERE_PROBE_SH_GROUP_SIZE];
    let mut samples = [Vec4::ZERO; SPHERE_PROBE_REMAP_GROUP_SIZE * SPHERE_PROBE_REMAP_GROUP_SIZE];
    let denominator = (extent - 2) as f32;

    for group_y in 0..groups_per_axis {
        for group_x in 0..groups_per_axis {
            for local_y in 0..SPHERE_PROBE_REMAP_GROUP_SIZE {
                for local_x in 0..SPHERE_PROBE_REMAP_GROUP_SIZE {
                    let local_index = local_y * SPHERE_PROBE_REMAP_GROUP_SIZE + local_x;
                    let column = group_x * SPHERE_PROBE_REMAP_GROUP_SIZE + local_x;
                    let row = group_y * SPHERE_PROBE_REMAP_GROUP_SIZE + local_y;
                    let weight = octahedral_texel_solid_angle(column, row, extent);
                    if weight == 0.0 {
                        samples[local_index] = Vec4::ZERO;
                        continue;
                    }
                    let direction =
                        octahedral_direction(column as f32 / denominator, row as f32 / denominator);
                    let radiance = brightness_clamp_max(
                        Vec3::from_array(cube_probe_sample(cubemap, face_texels, direction)),
                        sun_threshold,
                    );
                    samples[local_index] =
                        Vec4::new(radiance.x, radiance.y, radiance.z, radiance.z) * weight;
                }
            }

            let mut stride = samples.len() / 2;
            while stride != 0 {
                for index in 0..stride {
                    samples[index] += samples[index + stride];
                }
                stride /= 2;
            }

            let min_direction = octahedral_direction(
                (group_x * SPHERE_PROBE_REMAP_GROUP_SIZE) as f32 / denominator,
                (group_y * SPHERE_PROBE_REMAP_GROUP_SIZE) as f32 / denominator,
            )
            .normalize();
            let max_direction = octahedral_direction(
                ((group_x + 1) * SPHERE_PROBE_REMAP_GROUP_SIZE) as f32 / denominator,
                ((group_y + 1) * SPHERE_PROBE_REMAP_GROUP_SIZE) as f32 / denominator,
            )
            .normalize();
            let direction = (min_direction + max_direction).normalize();
            let signal = samples[0];
            let encoded = FastGiSh {
                l0_m0: signal * 0.282_094_8,
                l1_mn1: signal * (-0.488_602_52 * direction.y),
                l1_m0: signal * (0.488_602_52 * direction.z),
                l1_mp1: signal * (-0.488_602_52 * direction.x),
            };
            let group_index = group_y * groups_per_axis + group_x;
            let thread_sum = &mut sums_by_thread[group_index % SPHERE_PROBE_SH_GROUP_SIZE];
            thread_sum.l0_m0 += encoded.l0_m0;
            thread_sum.l1_mn1 += encoded.l1_mn1;
            thread_sum.l1_m0 += encoded.l1_m0;
            thread_sum.l1_mp1 += encoded.l1_mp1;
        }
    }

    let mut stride = SPHERE_PROBE_SH_GROUP_SIZE / 2;
    while stride != 0 {
        for index in 0..stride {
            let source = sums_by_thread[index + stride];
            sums_by_thread[index].l0_m0 += source.l0_m0;
            sums_by_thread[index].l1_mn1 += source.l1_mn1;
            sums_by_thread[index].l1_m0 += source.l1_m0;
            sums_by_thread[index].l1_mp1 += source.l1_mp1;
        }
        stride /= 2;
    }
    sums_by_thread[0]
}

fn cube_probe_sample(cubemap: &[EeveeProbeTexel], face_texels: usize, direction: Vec3) -> [f32; 3] {
    let extent = face_texels as f32;
    let (face, s, t) = cube_coordinates(direction.as_dvec3());
    let x = (f64::midpoint(s, 1.0) * f64::from(extent) - 0.5) as f32;
    let y = (f64::midpoint(t, 1.0) * f64::from(extent) - 0.5) as f32;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fraction_x = x - x.floor();
    let fraction_y = y - y.floor();
    let p00 = cube_probe_tap(cubemap, face_texels, face, x0, y0);
    let p10 = cube_probe_tap(cubemap, face_texels, face, x0 + 1, y0);
    let p01 = cube_probe_tap(cubemap, face_texels, face, x0, y0 + 1);
    let p11 = cube_probe_tap(cubemap, face_texels, face, x0 + 1, y0 + 1);
    std::array::from_fn(|channel| {
        let lower = p00[channel] * (1.0 - fraction_x) + p10[channel] * fraction_x;
        let upper = p01[channel] * (1.0 - fraction_x) + p11[channel] * fraction_x;
        lower * (1.0 - fraction_y) + upper * fraction_y
    })
}

fn cube_probe_tap(
    cubemap: &[EeveeProbeTexel],
    face_texels: usize,
    face: usize,
    column: i32,
    row: i32,
) -> [f32; 3] {
    let extent = face_texels as i32;
    if (0..extent).contains(&column) && (0..extent).contains(&row) {
        return load_eevee_probe_texel(
            cubemap[(face * face_texels + row as usize) * face_texels + column as usize],
        )
        .to_array();
    }
    let s = 2.0 * (column as f64 + 0.5) / f64::from(extent) - 1.0;
    let t = 2.0 * (row as f64 + 0.5) / f64::from(extent) - 1.0;
    let direction = face_direction(face, s, t);
    let (face, s, t) = cube_coordinates(direction);
    let column = (f64::midpoint(s, 1.0) * f64::from(extent)).floor() as usize;
    let row = (f64::midpoint(t, 1.0) * f64::from(extent)).floor() as usize;
    load_eevee_probe_texel(
        cubemap[(face * face_texels + row.min(face_texels - 1)) * face_texels
            + column.min(face_texels - 1)],
    )
    .to_array()
}

fn sample_world_probe(
    world: &World,
    groups: &Registry<NodeGroup>,
    images: &BTreeMap<String, ImageData>,
    face_texels: usize,
    mut capture_radiance: impl FnMut(DVec3) -> DVec3,
) -> Result<DVec3> {
    let mut workspace = EvaluatorWorkspace::default();
    let mut average_irradiance = DVec3::ZERO;
    let mut total_solid_angle = 0.0;
    for face in 0..6 {
        for row in 0..face_texels {
            for column in 0..face_texels {
                let direction = face_direction(
                    face,
                    texel_centre_at(column, face_texels),
                    texel_centre_at(row, face_texels),
                );
                let radiance = evaluate_world_with_workspace(
                    world,
                    groups,
                    images,
                    direction,
                    false,
                    &mut workspace,
                )?;
                let solid_angle = cube_texel_solid_angle_at(column, row, face_texels);
                let radiance = capture_radiance(radiance);
                average_irradiance +=
                    super::eevee_transmission::irradiance_probe_radiance(radiance) * solid_angle;
                total_solid_angle += solid_angle;
            }
        }
    }
    Ok(average_irradiance / total_solid_angle)
}

fn cube_texel_solid_angle_at(column: usize, row: usize, face_texels: usize) -> f64 {
    let edge = face_texels as f64;
    let s0 = 2.0 * column as f64 / edge - 1.0;
    let s1 = 2.0 * (column + 1) as f64 / edge - 1.0;
    let t0 = 2.0 * row as f64 / edge - 1.0;
    let t1 = 2.0 * (row + 1) as f64 / edge - 1.0;
    let area = |s: f64, t: f64| (s * t).atan2((1.0 + s * s + t * t).sqrt());
    area(s1, t1) - area(s0, t1) - area(s1, t0) + area(s0, t0)
}

fn texel_centre_at(index: usize, face_texels: usize) -> f64 {
    2.0 * (index as f64 + 0.5) / face_texels as f64 - 1.0
}

/// The direction through point `(s, t)` on cube face `face`: +X, -X, +Y, -Y, +Z, -Z in that order.
fn face_direction(face: usize, s: f64, t: f64) -> DVec3 {
    match face {
        0 => DVec3::new(1.0, -t, -s),
        1 => DVec3::new(-1.0, -t, s),
        2 => DVec3::new(s, 1.0, t),
        3 => DVec3::new(s, -1.0, -t),
        4 => DVec3::new(s, -t, 1.0),
        _ => DVec3::new(-s, -t, -1.0),
    }
}

/// The face and the `(s, t)` coordinates that `direction` pierces: the inverse of `face_direction`.
fn cube_coordinates(direction: DVec3) -> (usize, f64, f64) {
    let magnitude = direction.abs();
    if magnitude.x >= magnitude.y && magnitude.x >= magnitude.z {
        if direction.x > 0.0 {
            (0, -direction.z / magnitude.x, -direction.y / magnitude.x)
        } else {
            (1, direction.z / magnitude.x, -direction.y / magnitude.x)
        }
    } else if magnitude.y >= magnitude.z {
        if direction.y > 0.0 {
            (2, direction.x / magnitude.y, direction.z / magnitude.y)
        } else {
            (3, direction.x / magnitude.y, -direction.z / magnitude.y)
        }
    } else if direction.z > 0.0 {
        (4, direction.x / magnitude.z, -direction.y / magnitude.z)
    } else {
        (5, -direction.x / magnitude.z, -direction.y / magnitude.z)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Registry, World};

    #[test]
    fn cube_coordinates_invert_face_directions() {
        for face in 0..6 {
            for &s in &[-0.9, -0.3, 0.0, 0.4, 0.95] {
                for &t in &[-0.8, -0.1, 0.2, 0.7] {
                    let direction = face_direction(face, s, t).normalize();
                    let (found_face, found_s, found_t) = cube_coordinates(direction);
                    assert_eq!(found_face, face, "face of ({s}, {t})");
                    assert!(
                        (found_s - s).abs() < 1e-9,
                        "s of face {face}: {found_s} vs {s}"
                    );
                    assert!(
                        (found_t - t).abs() < 1e-9,
                        "t of face {face}: {found_t} vs {t}"
                    );
                }
            }
        }
    }

    #[test]
    fn eevee_cubemap_remap_filters_across_a_face_edge() {
        let cubemap: Vec<_> = (0..6 * FACE_TEXELS * FACE_TEXELS)
            .map(|index| {
                let face = index / (FACE_TEXELS * FACE_TEXELS);
                let row = index / FACE_TEXELS % FACE_TEXELS;
                let column = index % FACE_TEXELS;
                let direction = face_direction(
                    face,
                    texel_centre_at(column, FACE_TEXELS),
                    texel_centre_at(row, FACE_TEXELS),
                )
                .normalize();
                store_eevee_probe_texel(DVec3::new(
                    0.5 + direction.x * 0.25,
                    0.5 + direction.y * 0.25,
                    0.5 + direction.z * 0.25,
                ))
            })
            .collect();
        let to_vec3 = |direction: DVec3| {
            Vec3::new(direction.x as f32, direction.y as f32, direction.z as f32)
        };
        let from_left = sample_world_cubemap(
            &cubemap,
            FACE_TEXELS,
            to_vec3(face_direction(0, 1.0 - 1.0e-5, 0.23)),
        );
        let from_right = sample_world_cubemap(
            &cubemap,
            FACE_TEXELS,
            to_vec3(face_direction(0, 1.0 + 1.0e-5, 0.23)),
        );
        assert!(
            (from_left - from_right).abs().max_element() < 1.0e-3,
            "cube-edge samples differ: {from_left:?} vs {from_right:?}"
        );
    }

    #[test]
    fn octahedral_probe_texel_weights_cover_the_sphere() {
        let extent = 32;
        let mut total = 0.0_f32;
        for row in 0..extent {
            for column in 0..extent {
                total += octahedral_texel_solid_angle(column, row, extent);
            }
        }
        assert!((f64::from(total) - std::f64::consts::TAU * 2.0).abs() < 1.0e-4);
    }

    #[test]
    fn a_constant_world_is_its_colour_in_every_direction() -> Result<()> {
        let world = World {
            color: [0.2, 0.4, 0.6],
            strength: 2.0,
            ..World::default()
        };
        let probe = WorldProbe::build(
            &world,
            &Registry::default(),
            &BTreeMap::new(),
            WorldProbeMode::Path,
        )?;
        for direction in [DVec3::X, -DVec3::Y, DVec3::new(0.3, -0.5, 0.8).normalize()] {
            assert_eq!(probe.radiance(direction), DVec3::new(0.4, 0.8, 1.2));
        }
        assert!(probe.prefiltered_radiance(DVec3::X, 0.0).is_none());
        let expected =
            crate::render::eevee_transmission::irradiance_probe_radiance(DVec3::new(0.4, 0.8, 1.2));
        assert!((probe.average_irradiance() - expected).abs().max_element() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn eevee_prefilter_builder_stores_float16_mips_after_convolution() {
        let radiance = DVec3::new(0.2, 0.4, 0.6);
        let stored_radiance = store_eevee_probe_texel(radiance);
        let cubemap = vec![stored_radiance; 6 * FACE_TEXELS * FACE_TEXELS];
        let expected = load_eevee_probe_texel(stored_radiance).as_dvec3();
        assert_eq!(stored_radiance[3].to_f32(), 1.0);
        let (mips, _) = build_eevee_prefiltered_atlas(&cubemap, FACE_TEXELS, 10.0, 1.0e20, false);
        for (lod, mip) in mips.iter().enumerate() {
            let extent = eevee_probe_mip_extent(FACE_TEXELS, lod);
            assert_eq!(mip.len(), extent * extent);
            for row in 0..extent {
                for column in 0..extent {
                    let stored = load_eevee_probe_texel(mip[row * extent + column]).as_dvec3();
                    assert!(
                        (stored - expected).abs().max_element() < 1.0e-3,
                        "constant probe changed at mip texel ({column}, {row})"
                    );
                }
            }
        }
    }
    #[test]
    fn eevee_prefilter_applies_surface_indirect_clamp_after_world_sun_threshold() {
        let cubemap =
            vec![store_eevee_probe_texel(DVec3::splat(100.0)); 6 * FACE_TEXELS * FACE_TEXELS];
        let (mips, _) = build_eevee_prefiltered_atlas(&cubemap, FACE_TEXELS, 50.0, 5.0, false);
        let center = eevee_probe_mip_extent(FACE_TEXELS, 0) / 2;
        let stored = load_eevee_probe_texel(mips[0][center * center + center]);
        assert_eq!(stored, Vec3::splat(5.0));
        assert_eq!(eevee_probe_clamp_surface_indirect(0.0), 1.0e20);
    }
    #[test]
    fn eevee_prefilter_mixes_the_source_roughness_lods() -> Result<(), String> {
        let face_texels = FACE_TEXELS;
        let mips = std::array::from_fn(|mip| {
            let extent = eevee_probe_mip_extent(face_texels, mip);
            vec![store_eevee_probe_texel(DVec3::splat(mip as f64)); extent * extent]
        });
        let probe = WorldProbe {
            radiance: WorldProbeRadiance::Eevee {
                cubemap: Vec::new(),
                sun_threshold: 10.0,
                surface_indirect_clamp: 1.0e20,
            },
            face_texels,
            eevee_prefiltered: Some(mips),
            average_irradiance: DVec3::ZERO,
            world_sun: None,
            fast_gi_sh: OnceLock::new(),
        };
        let roughness = 0.35_f64;
        let source_lod =
            (0.5 + (0.5_f32.sqrt() - 0.5) * 0.4) * (EEVEE_PROBE_MIPMAP_LEVELS - 1) as f32;
        let sampled = probe
            .prefiltered_radiance(DVec3::Z, roughness)
            .ok_or_else(|| "the EEVEE atlas is present".to_owned())?;
        assert!((sampled.x - f64::from(source_lod)).abs() < 1.0e-6);
        let high_roughness = probe
            .prefiltered_radiance(DVec3::Z, 0.7)
            .ok_or_else(|| "the EEVEE atlas is present".to_owned())?;
        assert_eq!(high_roughness, DVec3::splat(4.0));
        Ok(())
    }

    #[test]
    fn eevee_capture_extents_follow_the_scene_resolution() {
        let mode = WorldProbeMode::Eevee {
            cubemap_resolution: 512,
            surface_indirect_clamp: 10.0,
        };
        let face_texels = mode.face_texels();
        assert_eq!(face_texels, 256);
        assert_eq!(
            std::array::from_fn::<_, EEVEE_PROBE_MIPMAP_LEVELS, _>(|mip| {
                eevee_probe_mip_extent(face_texels, mip)
            }),
            [512, 256, 128, 64, 32]
        );
        assert_eq!(WorldProbeMode::Path.face_texels(), FACE_TEXELS);
    }

    #[test]
    fn eevee_world_sun_uses_thresholded_captured_radiance() -> Result<(), String> {
        let build_probe = |color, threshold, surface_indirect_clamp| {
            let world = World {
                color: [color; 3],
                eevee_sun_threshold: Some(threshold),
                ..World::default()
            };
            WorldProbe::build(
                &world,
                &Registry::default(),
                &BTreeMap::new(),
                WorldProbeMode::Eevee {
                    cubemap_resolution: 128,
                    surface_indirect_clamp,
                },
            )
        };
        let bright = build_probe(12.0, 10.0, 10.0).map_err(|error| error.to_string())?;
        let sun = bright.world_sun().ok_or_else(|| {
            "bright world radiance above threshold should extract a WorldSun".to_owned()
        })?;
        let expected = 8.0 * std::f32::consts::PI;
        assert!((sun.radiance - Vec3::splat(expected)).abs().max_element() < 1.0e-2);
        assert!(sun.direction.is_finite());
        let limited = build_probe(12.0, 10.0, 5.0).map_err(|error| error.to_string())?;
        let limited_sun = limited.world_sun().ok_or_else(|| {
            "surface indirect clamp should not prevent extracting a WorldSun".to_owned()
        })?;
        assert_eq!(limited_sun.radiance, sun.radiance);
        assert!(limited.average_irradiance().max_element() <= 5.0);
        assert!(
            limited
                .prefiltered_radiance(DVec3::Z, 0.0)
                .is_some_and(|radiance| radiance.max_element() <= 5.0)
        );
        assert!(
            build_probe(9.0, 10.0, 10.0)
                .map_err(|error| error.to_string())?
                .world_sun()
                .is_none()
        );
        assert!(
            build_probe(12.0, 0.0, 10.0)
                .map_err(|error| error.to_string())?
                .world_sun()
                .is_none()
        );
        Ok(())
    }
    #[test]
    fn eevee_world_sun_keeps_excess_with_safe_direction_fallback() -> Result<(), String> {
        for direction_sum in [
            Vec4::ZERO,
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0e-18, 0.0, 0.0, 1.0),
            Vec4::new(f32::NAN, 0.0, 0.0, 1.0),
        ] {
            let sun = eevee_world_sun_from_reduction(Vec3::splat(2.0), direction_sum)
                .ok_or_else(|| "positive excess radiance should produce a WorldSun".to_owned())?;
            assert_eq!(sun.radiance, Vec3::splat(2.0));
            assert_eq!(sun.direction, Vec3::X);
        }
        Ok(())
    }

    #[test]
    fn volume_sh_reduction_keeps_all_remap_groups() {
        let groups = vec![[1.0, 2.0, 3.0]; SPHERE_PROBE_SH_GROUP_SIZE + 1];
        assert_eq!(
            sum_sphere_probe_coefficients(&groups),
            [
                groups.len() as f32,
                2.0 * groups.len() as f32,
                3.0 * groups.len() as f32
            ]
        );
    }
    #[test]
    fn eevee_prefilter_is_continuous_across_an_octahedral_edge() -> Result<(), String> {
        let face_texels = FACE_TEXELS;
        let mips = std::array::from_fn(|mip| {
            let extent = eevee_probe_mip_extent(face_texels, mip);
            let denominator = (extent - 2) as f32;
            (0..extent * extent)
                .map(|index| {
                    let column = index % extent;
                    let row = index / extent;
                    let direction =
                        octahedral_direction(column as f32 / denominator, row as f32 / denominator)
                            .normalize();
                    store_eevee_probe_texel(DVec3::new(
                        0.5 + f64::from(direction.x) * 0.25,
                        0.5 + f64::from(direction.y) * 0.25,
                        0.5 + f64::from(direction.z) * 0.25,
                    ))
                })
                .collect()
        });
        let probe = WorldProbe {
            radiance: WorldProbeRadiance::Eevee {
                cubemap: Vec::new(),
                sun_threshold: 10.0,
                surface_indirect_clamp: 1.0e20,
            },
            face_texels,
            eevee_prefiltered: Some(mips),
            average_irradiance: DVec3::ZERO,
            fast_gi_sh: OnceLock::new(),
            world_sun: None,
        };
        let from_left = probe
            .prefiltered_radiance(DVec3::new(-1.0e-5, 1.0, -1.0), 0.0)
            .ok_or_else(|| "the EEVEE atlas is present".to_owned())?;
        let from_right = probe
            .prefiltered_radiance(DVec3::new(1.0e-5, 1.0, -1.0), 0.0)
            .ok_or_else(|| "the EEVEE atlas is present".to_owned())?;
        assert!(
            (from_left - from_right).abs().max_element() < 1.0e-4,
            "octahedral seam samples differ: {from_left:?} vs {from_right:?}"
        );
        Ok(())
    }
}
