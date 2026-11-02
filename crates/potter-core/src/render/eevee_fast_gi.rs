use crate::error::{ErrorCode, PotError, Result};
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

const TILE_SIZE: u32 = 8;
const PI: f32 = std::f32::consts::PI;
const TAU: f32 = std::f32::consts::TAU;
const SH_L0_COEF: f32 = 0.282_094_8;
const SH_L1_COEF: f32 = 0.488_602_52;
const SH_SQRT3: f32 = 1.732_050_8;

#[derive(Clone, Copy)]
pub(crate) struct FastGiTexture<'a, T> {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: &'a [T],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FastGiMethod {
    Full,
    AoOnly,
}

#[derive(Clone, Copy)]
pub(crate) struct FastGiSettings {
    /// Blender `fast_gi_resolution` and ray tracing `resolution_scale` enum values.
    pub(crate) resolution: u32,
    pub(crate) raytrace_resolution: u32,
    pub(crate) ray_count: u32,
    pub(crate) step_count: u32,
    pub(crate) quality: f32,
    pub(crate) bias: f32,
    /// Blender `fast_gi_thickness_near`.
    pub(crate) thickness_near: f32,
    /// Blender `fast_gi_distance`, with zero using twice the camera frustum bound radius.
    pub(crate) distance: f32,
    pub(crate) camera_bound_radius: f32,
    pub(crate) method: FastGiMethod,
    pub(crate) trace_max_roughness: f32,
    pub(crate) backface_hit_enabled: bool,
    pub(crate) backface_radiance_scale: f32,
    pub(crate) surface_indirect_clamp: f32,
    pub(crate) hiz_uv_scale: Vec2,
    pub(crate) sample_index: u32,
    pub(crate) sampling_ao: Vec3,
}

#[derive(Clone, Copy)]
pub(crate) struct FastGiView {
    pub(crate) view: Mat4,
    pub(crate) view_inverse: Mat4,
    pub(crate) projection: Mat4,
    pub(crate) projection_inverse: Mat4,
}

#[derive(Clone, Copy)]
pub(crate) struct FastGiSh {
    pub(crate) l0_m0: Vec4,
    pub(crate) l1_mn1: Vec4,
    pub(crate) l1_m0: Vec4,
    pub(crate) l1_mp1: Vec4,
}

impl FastGiSh {
    pub(crate) const ZERO: Self = Self {
        l0_m0: Vec4::ZERO,
        l1_mn1: Vec4::ZERO,
        l1_m0: Vec4::ZERO,
        l1_mp1: Vec4::ZERO,
    };

    fn encode(&mut self, direction: Vec3, amplitude: Vec4) {
        self.l0_m0 += amplitude * SH_L0_COEF;
        self.l1_mn1 += amplitude * (-SH_L1_COEF * direction.y);
        self.l1_m0 += amplitude * (SH_L1_COEF * direction.z);
        self.l1_mp1 += amplitude * (-SH_L1_COEF * direction.x);
    }

    pub(super) fn evaluate_lambert(self, direction: Vec3) -> Vec4 {
        let l1 = self.l1_mn1 * (-SH_L1_COEF * direction.y)
            + self.l1_m0 * (SH_L1_COEF * direction.z)
            + self.l1_mp1 * (-SH_L1_COEF * direction.x);
        (self.l0_m0 * SH_L0_COEF + l1 * (2.0 / 3.0)).max(Vec4::ZERO)
    }

    fn madd(self, weight: f32, add: Self) -> Self {
        Self {
            l0_m0: self.l0_m0 * weight + add.l0_m0,
            l1_mn1: self.l1_mn1 * weight + add.l1_mn1,
            l1_m0: self.l1_m0 * weight + add.l1_m0,
            l1_mp1: self.l1_mp1 * weight + add.l1_mp1,
        }
    }

    fn scale(self, weight: f32) -> Self {
        Self {
            l0_m0: self.l0_m0 * weight,
            l1_mn1: self.l1_mn1 * weight,
            l1_m0: self.l1_m0 * weight,
            l1_mp1: self.l1_mp1 * weight,
        }
    }

    fn scale_channels(self, weight: Vec4) -> Self {
        Self {
            l0_m0: self.l0_m0 * weight,
            l1_mn1: self.l1_mn1 * weight,
            l1_m0: self.l1_m0 * weight,
            l1_mp1: self.l1_mp1 * weight,
        }
    }

    fn rotate_view_to_world(mut self, view_inverse: Mat4) -> Self {
        let rotation = Mat3::from_mat4(view_inverse);
        let input = [self.l1_mp1, self.l1_mn1, -self.l1_m0];
        let mut output = [Vec4::ZERO; 3];
        for channel in 0..4 {
            let value =
                rotation * Vec3::new(input[0][channel], input[1][channel], input[2][channel]);
            output[0][channel] = value.x;
            output[1][channel] = value.y;
            output[2][channel] = value.z;
        }
        self.l1_mn1 = output[1];
        self.l1_m0 = -output[2];
        self.l1_mp1 = output[0];
        self
    }

    fn clamp_energy(self, clamp_value: f32) -> Self {
        let magnitude = |l0: f32, mn1: f32, m0: f32, mp1: f32| {
            l0.abs() * SH_L0_COEF + Vec3::new(mn1, m0, mp1).length() * SH_L1_COEF
        };
        let maximum = magnitude(self.l0_m0.x, self.l1_mn1.x, self.l1_m0.x, self.l1_mp1.x)
            .max(magnitude(
                self.l0_m0.y,
                self.l1_mn1.y,
                self.l1_m0.y,
                self.l1_mp1.y,
            ))
            .max(magnitude(
                self.l0_m0.z,
                self.l1_mn1.z,
                self.l1_m0.z,
                self.l1_mp1.z,
            ));
        let factor = clamp_value * safe_rcp(maximum);
        if factor < 1.0 {
            self.scale(factor)
        } else {
            self
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct FastGiResolveClosure {
    pub(crate) bin_index: u8,
    pub(crate) roughness: f32,
    /// Source `bxdf_lightprobe_ray(...).dominant_direction`, in world space.
    pub(crate) dominant_direction: Vec3,
    pub(crate) transmission: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct FastGiResolvePixel {
    /// Actual volume-probe SH loaded for this pixel; required when resolving a `GBuffer` closure.
    pub(crate) volume_irradiance: Option<FastGiSh>,
    pub(crate) closures: [Option<FastGiResolveClosure>; 3],
}

pub(crate) struct FastGiInput<'a> {
    /// All image slices use EEVEE texture texel order (`x + y * width`, lower-left origin).
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Raw hardware depth; Blender's reverse-Z read is `1.0 - depth`.
    pub(crate) depth: &'a [f32],
    /// `HiZ` values are already converted back from reverse-Z by Blender's `HiZ` build.
    pub(crate) hiz: &'a [FastGiTexture<'a, f32>],
    /// Previous-frame radiance feedback texture, stored as `SFLOAT_16_16_16_16`.
    pub(crate) history_radiance: FastGiTexture<'a, [f32; 4]>,
    pub(crate) history_valid: bool,
    /// Current world-space `GBuffer` normals and the `GBuffer` non-empty mask.
    pub(crate) world_normals: &'a [[f32; 3]],
    pub(crate) processed: &'a [bool],
    /// Actual utility blue-noise layer and source per-frame AO sampling values.
    pub(crate) blue_noise: FastGiTexture<'a, [f32; 4]>,
    /// Three existing closure radiance images in `RAYTRACE_RADIANCE_FORMAT`.
    pub(crate) closures: [FastGiTexture<'a, [f32; 3]>; 3],
    pub(crate) resolve: &'a [FastGiResolvePixel],
    pub(crate) view: FastGiView,
    pub(crate) history_view_projection: Mat4,
    pub(crate) settings: FastGiSettings,
}

pub(crate) struct FastGiShTexture {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Source `SFLOAT_16_16_16_16` L0.M0 image.
    pub(crate) l0_m0: Vec<[u16; 4]>,
    /// Source `UNORM_8_8_8_8` L1.Mn1, L1.M0 and L1.Mp1 images.
    pub(crate) l1_mn1: Vec<[u8; 4]>,
    pub(crate) l1_m0: Vec<[u8; 4]>,
    pub(crate) l1_mp1: Vec<[u8; 4]>,
}

impl FastGiShTexture {
    pub(crate) fn load(&self, x: u32, y: u32) -> Option<FastGiSh> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let index = y as usize * self.width as usize + x as usize;
        let l0_m0 = Vec4::from_array(self.l0_m0.get(index)?.map(f16_to_f32));
        let factor = l0_m0 * SH_SQRT3;
        let decompress = |value: [u8; 4]| {
            (Vec4::from_array(value.map(|byte| f32::from(byte) / 255.0)) * 2.0 - Vec4::ONE) * factor
        };
        Some(FastGiSh {
            l0_m0,
            l1_mn1: decompress(*self.l1_mn1.get(index)?),
            l1_m0: decompress(*self.l1_m0.get(index)?),
            l1_mp1: decompress(*self.l1_mp1.get(index)?),
        })
    }
}

pub(crate) struct FastGiImage<T> {
    pub(crate) pixels: Vec<T>,
}

pub(crate) struct FastGiOutput {
    /// Three source-format closure radiance images after read/modify/write resolve.
    pub(crate) closure_radiance: [FastGiImage<[f32; 3]>; 3],
}

pub(crate) fn fast_gi_should_run(enabled: bool, trace_max_roughness: f32) -> bool {
    enabled && trace_max_roughness < 1.0
}

pub(crate) fn run_fast_gi(input: &FastGiInput<'_>) -> Result<FastGiOutput> {
    let width = input.width;
    let height = input.height;
    let pixel_count = checked_pixel_count(width, height)?;
    if input.hiz.is_empty() {
        return Err(PotError::invalid_argument(
            "FastGI requires the source HiZ mip chain",
        ));
    }
    validate_texture(&input.history_radiance)?;
    validate_texture(&input.blue_noise)?;
    for mip in input.hiz {
        validate_texture(mip)?;
    }
    if input.hiz[0].width < width
        || input.hiz[0].height < height
        || input.depth.len() != pixel_count
        || input.world_normals.len() != pixel_count
        || input.processed.len() != pixel_count
        || input.resolve.len() != pixel_count
    {
        return Err(PotError::invalid_argument(
            "FastGI full-resolution buffers must match the source render extent",
        ));
    }
    for closure in input.closures {
        validate_texture(&closure)?;
        if closure.width != width || closure.height != height {
            return Err(PotError::invalid_argument(
                "FastGI closure images must match the source render extent",
            ));
        }
    }
    if !(1..=16).contains(&input.settings.ray_count)
        || !(1..=64).contains(&input.settings.step_count)
        || !(0.0..=1.0).contains(&input.settings.quality)
        || !(0.0..=1.0).contains(&input.settings.bias)
    {
        return Err(PotError::invalid_argument(
            "FastGI settings are outside Blender's source ranges",
        ));
    }

    let settings = input.settings;
    let scale = settings.resolution.max(1).next_power_of_two();
    let raytrace_scale = settings.raytrace_resolution.max(1).next_power_of_two();
    let tracing_width = ceil_to_multiple(div_ceil(width, scale), TILE_SIZE);
    let tracing_height = ceil_to_multiple(div_ceil(height, scale), TILE_SIZE);
    checked_pixel_count(tracing_width, tracing_height)?;
    let resolution_bias = random_in_tile(settings.sample_index, scale);
    let fast_gi_lod_bias = raytrace_scale.ilog2() as f32;
    let fast_gi_uv_scale = Vec2::new(
        width as f32 / (tracing_width * scale) as f32,
        height as f32 / (tracing_height * scale) as f32,
    );
    let lod_factor = (1.0 - settings.quality) * 0.2;
    let angle_bias = 1.0 / (1.0e-8_f32).max(1.0 - settings.bias);
    let search_distance = if settings.distance > 0.0 {
        settings.distance
    } else {
        settings.camera_bound_radius * 2.0
    };
    let backface_hit_scale = if settings.backface_hit_enabled {
        settings.backface_radiance_scale
    } else {
        0.0
    };
    let pixel_size = Vec2::new(2.0 / width as f32, 2.0 / height as f32);
    let thickness = build_thickness_parameters(
        input.view.projection,
        width,
        height,
        4.0 * (64.0 / settings.step_count as f32).sqrt(),
        settings.thickness_near,
    );

    let (radiance_mips, normal_mips) =
        setup(input, tracing_width, tracing_height, scale, resolution_bias)?;
    let mut scanned = FastGiShTexture::zeroed(tracing_width, tracing_height)?;
    for y in 0..tracing_height {
        for x in 0..tracing_width {
            let output_index = y as usize * tracing_width as usize + x as usize;
            let full_x = (x * scale + resolution_bias[0]).min(width - 1);
            let full_y = (y * scale + resolution_bias[1]).min(height - 1);
            let full_index = full_y as usize * width as usize + full_x as usize;
            if !input.processed[full_index] {
                continue;
            }
            let uv = (Vec2::new(full_x as f32, full_y as f32) + Vec2::splat(0.5))
                / Vec2::new(width as f32, height as f32);
            let position = point_screen_to_view(
                input.view,
                uv.extend(hiz_fetch(input.hiz, full_x, full_y, 0)),
            );
            let normal = unpack_normal(normal_mips[0].pixels[output_index]);
            let view_normal = Vec3::new(normal[0], normal[1], normal[2]) * 2.0 - Vec3::ONE;
            let blue = sample_nearest(input.blue_noise, Vec2::new(x as f32, y as f32));
            let noise = Vec4::new(
                fract(blue[0] + settings.sampling_ao.x),
                fract(blue[1] + settings.sampling_ao.y),
                fract(blue[2] + settings.sampling_ao.z),
                fract(blue[3] + settings.sampling_ao.x),
            );
            let sh = eval_horizon(
                input,
                position,
                view_normal,
                noise,
                pixel_size,
                search_distance,
                angle_bias,
                lod_factor,
                settings.ray_count as i32,
                settings.step_count as i32,
                backface_hit_scale,
                fast_gi_uv_scale,
                fast_gi_lod_bias,
                thickness,
                &radiance_mips,
                &normal_mips,
            );
            scanned.store(output_index, sh);
        }
    }

    let mut denoised = FastGiShTexture::zeroed(tracing_width, tracing_height)?;
    let texel_size = Vec2::new(1.0 / tracing_width as f32, 1.0 / tracing_height as f32);
    for y in 0..tracing_height {
        for x in 0..tracing_width {
            let index = y as usize * tracing_width as usize + x as usize;
            let full_x = x * scale + resolution_bias[0];
            let full_y = y * scale + resolution_bias[1];
            let normal = unpack_normal(normal_mips[0].pixels[index]);
            if normal[3] == 0.0 {
                continue;
            }
            let center_depth = hiz_fetch(input.hiz, full_x, full_y, 0);
            let center_uv = Vec2::new(x as f32, y as f32) * texel_size;
            let center_position = point_screen_to_world(input.view, center_uv.extend(center_depth));
            let center_normal = view_normal_to_world(
                input.view,
                Vec3::new(normal[0], normal[1], normal[2]) * 2.0 - Vec3::ONE,
            );
            let mut sum = FastGiSh::ZERO;
            let mut sum_weight = 0.0;
            for offset_y in -1..=1 {
                for offset_x in -1..=1 {
                    let sample_x = x as i32 + offset_x;
                    let sample_y = y as i32 + offset_y;
                    if sample_x < 0
                        || sample_y < 0
                        || sample_x >= tracing_width as i32
                        || sample_y >= tracing_height as i32
                    {
                        continue;
                    }
                    let sample_x = sample_x as u32;
                    let sample_y = sample_y as u32;
                    let sample_index =
                        sample_y as usize * tracing_width as usize + sample_x as usize;
                    let sample_normal_encoded = unpack_normal(normal_mips[0].pixels[sample_index]);
                    if sample_normal_encoded[3] == 0.0 {
                        continue;
                    }
                    let sample_full_x = sample_x * scale + resolution_bias[0];
                    let sample_full_y = sample_y * scale + resolution_bias[1];
                    let sample_depth = hiz_fetch(input.hiz, sample_full_x, sample_full_y, 0);
                    let sample_uv = (Vec2::new(sample_x as f32, sample_y as f32)
                        + Vec2::splat(0.5))
                        * texel_size;
                    let sample_position =
                        point_screen_to_world(input.view, sample_uv.extend(sample_depth));
                    let sample_normal = view_normal_to_world(
                        input.view,
                        Vec3::new(
                            sample_normal_encoded[0],
                            sample_normal_encoded[1],
                            sample_normal_encoded[2],
                        ) * 2.0
                            - Vec3::ONE,
                    );
                    let depth_weight =
                        planar_weight(center_normal, center_position, sample_position, 100.0);
                    let spatial_weight = gaussian_weight(
                        gaussian_factor(1.5, 1.5),
                        (offset_x * offset_x + offset_y * offset_y) as f32,
                    );
                    let normal_weight = angle_weight(center_normal, sample_normal);
                    let weight = (depth_weight * spatial_weight * normal_weight).max(1.0e-6);
                    if let Some(sample_sh) = scanned.load(sample_x, sample_y) {
                        sum = sample_sh.madd(weight, sum);
                        sum_weight += weight;
                    }
                }
            }
            let resolved = sum.scale(safe_rcp(sum_weight));
            denoised.store(index, resolved);
        }
    }

    let mut closure_radiance = [
        copy_radiance(input.closures[0])?,
        copy_radiance(input.closures[1])?,
        copy_radiance(input.closures[2])?,
    ];
    resolve(
        input,
        scale,
        resolution_bias,
        &normal_mips[0],
        &denoised,
        &mut closure_radiance,
    )?;
    Ok(FastGiOutput { closure_radiance })
}

impl FastGiShTexture {
    fn zeroed(width: u32, height: u32) -> Result<Self> {
        let count = checked_pixel_count(width, height)?;
        Ok(Self {
            width,
            height,
            l0_m0: allocate(count, [0_u16; 4], "FastGI SH L0")?,
            l1_mn1: allocate(count, [128_u8; 4], "FastGI SH L1")?,
            l1_m0: allocate(count, [128_u8; 4], "FastGI SH L1")?,
            l1_mp1: allocate(count, [128_u8; 4], "FastGI SH L1")?,
        })
    }

    fn store(&mut self, index: usize, sh: FastGiSh) {
        let compressed = sh_compress(sh);
        self.l0_m0[index] = compressed.l0_m0.to_array().map(f32_to_f16);
        self.l1_mn1[index] = compressed.l1_mn1.to_array().map(unorm8);
        self.l1_m0[index] = compressed.l1_m0.to_array().map(unorm8);
        self.l1_mp1[index] = compressed.l1_mp1.to_array().map(unorm8);
    }
}

struct SetupRadianceMip {
    width: u32,
    height: u32,
    pixels: Vec<u32>,
}

struct SetupNormalMip {
    width: u32,
    height: u32,
    pixels: Vec<u32>,
}

fn setup(
    input: &FastGiInput<'_>,
    width: u32,
    height: u32,
    scale: u32,
    bias: [u32; 2],
) -> Result<([SetupRadianceMip; 4], [SetupNormalMip; 4])> {
    let count = checked_pixel_count(width, height)?;
    let mut radiance_values = allocate(count, Vec3::ZERO, "FastGI setup radiance")?;
    let mut normal_values = allocate(count, Vec3::ZERO, "FastGI setup normals")?;
    for y in 0..height {
        for x in 0..width {
            let index = y as usize * width as usize + x as usize;
            let full_x = (x * scale + bias[0]).min(input.width - 1);
            let full_y = (y * scale + bias[1]).min(input.height - 1);
            let full_index = full_y as usize * input.width as usize + full_x as usize;
            let normal = if input.processed[full_index] {
                normal_world_to_view(
                    input.view,
                    Vec3::from_array(input.world_normals[full_index]),
                )
            } else {
                Vec3::Z
            };
            normal_values[index] = normal * 0.5 + Vec3::splat(0.5);
            let uv = (Vec2::new(full_x as f32, full_y as f32) + Vec2::splat(0.5))
                / Vec2::new(input.width as f32, input.height as f32);
            let depth = 1.0 - input.depth[full_index];
            let world_position = point_screen_to_world(input.view, uv.extend(depth));
            let previous_clip = input.history_view_projection * world_position.extend(1.0);
            let previous_uv = (previous_clip.truncate() / previous_clip.w) * 0.5 + Vec3::splat(0.5);
            let radiance = if input.history_valid {
                sample_linear_rgba16(input.history_radiance, previous_uv.truncate()).truncate()
            } else {
                Vec3::ZERO
            };
            let clamp = input.settings.surface_indirect_clamp;
            radiance_values[index] =
                radiance * (clamp / radiance.abs().max_element().max(1.0e-8)).clamp(0.0, 1.0);
        }
    }

    let mut radiance_mips = Vec::new();
    let mut normal_mips = Vec::new();
    radiance_mips
        .try_reserve_exact(4)
        .map_err(|_| memory_error("FastGI radiance mips"))?;
    normal_mips
        .try_reserve_exact(4)
        .map_err(|_| memory_error("FastGI normal mips"))?;
    for lod in 0..4_u32 {
        let mip_width = width >> lod;
        let mip_height = height >> lod;
        let mip_count = checked_pixel_count(mip_width, mip_height)?;
        let mut radiance = allocate(mip_count, 0_u32, "FastGI radiance mip")?;
        let mut normal = allocate(mip_count, 0_u32, "FastGI normal mip")?;
        for y in 0..mip_height {
            for x in 0..mip_width {
                let index = y as usize * mip_width as usize + x as usize;
                radiance[index] = pack_ufloat_r11g11b10(radiance_values[index]);
                let alpha = if lod == 0 {
                    let full_x = (x * scale + bias[0]).min(input.width - 1);
                    let full_y = (y * scale + bias[1]).min(input.height - 1);
                    if input.processed[(full_y * input.width + full_x) as usize] {
                        3
                    } else {
                        0
                    }
                } else {
                    0
                };
                normal[index] = pack_unorm1010102(normal_values[index], alpha);
            }
        }
        radiance_mips.push(SetupRadianceMip {
            width: mip_width,
            height: mip_height,
            pixels: radiance,
        });
        normal_mips.push(SetupNormalMip {
            width: mip_width,
            height: mip_height,
            pixels: normal,
        });
        if lod < 3 {
            let next_width = width >> (lod + 1);
            let next_height = height >> (lod + 1);
            let mut next_radiance = allocate(
                checked_pixel_count(next_width, next_height)?,
                Vec3::ZERO,
                "FastGI setup radiance mip",
            )?;
            let mut next_normal = allocate(
                checked_pixel_count(next_width, next_height)?,
                Vec3::ZERO,
                "FastGI setup normal mip",
            )?;
            for y in 0..next_height {
                for x in 0..next_width {
                    let mut radiance_average = Vec3::ZERO;
                    let mut normal_average = Vec3::ZERO;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let source_index = ((y * 2 + dy) as usize) * mip_width as usize
                                + (x * 2 + dx) as usize;
                            radiance_average += radiance_values[source_index];
                            normal_average += normal_values[source_index];
                        }
                    }
                    let index = y as usize * next_width as usize + x as usize;
                    next_radiance[index] = radiance_average * 0.25;
                    next_normal[index] = normal_average * 0.25;
                }
            }
            radiance_values = next_radiance;
            normal_values = next_normal;
        }
    }
    let radiance_mips: [SetupRadianceMip; 4] = radiance_mips
        .try_into()
        .map_err(|_| PotError::invalid_operation("FastGI setup must produce four radiance mips"))?;
    let normal_mips: [SetupNormalMip; 4] = normal_mips
        .try_into()
        .map_err(|_| PotError::invalid_operation("FastGI setup must produce four normal mips"))?;
    Ok((radiance_mips, normal_mips))
}

fn eval_horizon(
    input: &FastGiInput<'_>,
    position: Vec3,
    normal: Vec3,
    noise: Vec4,
    pixel_size: Vec2,
    search_distance: f32,
    angle_bias: f32,
    lod_factor: f32,
    slice_count: i32,
    sample_count: i32,
    backface_hit_scale: f32,
    uv_scale: Vec2,
    lod_bias: f32,
    thickness: ScreenThicknessParameters,
    radiance_mips: &[SetupRadianceMip; 4],
    normal_mips: &[SetupNormalMip; 4],
) -> FastGiSh {
    let view_direction = if input.view.projection.w_axis.w == 0.0 {
        (-position).normalize()
    } else {
        Vec3::Z
    };
    let mut direction = Vec2::ZERO;
    if slice_count <= 2 {
        direction = sample_circle(noise.x / (2 * slice_count) as f32);
    }
    let mut total_length = 0.0;
    let mut accumulated = FastGiSh::ZERO;
    for slice in 0..slice_count {
        if slice_count > 2 {
            direction = sample_circle((slice as f32 + noise.x) / (2 * slice_count) as f32);
        }
        let bitangent = view_direction.cross(direction.extend(0.0)).normalize();
        let tangent = bitangent.cross(view_direction);
        let projected = normal - bitangent * normal.dot(bitangent);
        let mut projected_length = 0.0;
        let projected_normal = safe_normalize_and_length(projected, &mut projected_length);
        let normal_sine = projected_normal.dot(tangent);
        let normal_cosine = projected_normal.dot(view_direction);
        let mut normal_angle = sign(normal_sine) * acos_fast(normal_cosine);
        normal_angle += (noise.z - 0.5) * (PI / 32.0) * angle_bias;
        let mut slice_sh = FastGiSh::ZERO;
        slice_sh.encode(normal, Vec4::W);
        let mut slice_mask = 0_u32;
        for side in 0..2 {
            let ray_direction = if side == 0 { direction } else { -direction };
            let ray = screen_space_ray(
                input.view.projection,
                position,
                ray_direction,
                search_distance,
                pixel_size,
            );
            let mut thickness_estimator = ScreenThicknessEstimator::new(position.z);
            for step in 0..sample_count {
                let sample_position = (step as f32 + noise.y) / sample_count as f32;
                let time = 1.0 + sample_position * sample_position * ray.max_time;
                let lod = (time * lod_factor).log2();
                let screen_uv = ray.origin + ray.direction * time;
                let depth_lod = (noise.w + lod).floor() as i32;
                let sample_depth = sample_hiz(
                    input.hiz,
                    screen_uv * input.settings.hiz_uv_scale,
                    depth_lod,
                );
                if sample_depth.partial_cmp(&1.0) == Some(std::cmp::Ordering::Equal) {
                    continue;
                }
                let sample_view_z =
                    point_screen_to_view(input.view, Vec3::new(0.0, 0.0, sample_depth)).z;
                let min_thickness = thickness.pixel_depth_thickness_at(sample_view_z);
                let sample_thickness =
                    thickness_estimator.thickness(sample_depth, time, min_thickness);
                let front_depth = sample_depth + 2.0 * 2.4e-7;
                let back_depth = sample_depth + sample_thickness;
                let front_position =
                    point_screen_to_view(input.view, screen_uv.extend(front_depth));
                let back_position = point_screen_to_view(input.view, screen_uv.extend(back_depth));
                let local_front = (front_position - position) / search_distance;
                let local_back = (back_position - position) / search_distance;
                let max_z =
                    sqrt_fast((1.0 - local_front.truncate().length_squared()).clamp(0.0, 1.0));
                let mut local_front = local_front;
                let mut local_back = local_back;
                local_front.z = local_front.z.clamp(-max_z, max_z);
                local_back.z = local_back.z.clamp(-max_z, max_z);
                if local_front.z.partial_cmp(&local_back.z) == Some(std::cmp::Ordering::Equal) {
                    continue;
                }
                let light_front = local_front.normalize();
                let light_back = local_back.normalize();
                let view_front = light_front.dot(view_direction);
                let view_back = light_back.dot(view_direction);
                let theta_front = acos_fast(view_front);
                let theta_back = acos_fast(view_back).max(theta_front);
                let theta = if side == 0 {
                    Vec2::new(theta_front, theta_back)
                } else {
                    Vec2::new(-theta_back, -theta_front)
                };
                let sample_lod = lod - lod_bias;
                let radiance_uv = screen_uv * uv_scale;
                let incoming = sample_radiance_mip(radiance_mips, radiance_uv, sample_lod);
                let sample_normal = sample_normal_mip(normal_mips, radiance_uv, sample_lod);
                let mut radiance = if input.settings.method == FastGiMethod::AoOnly {
                    Vec3::ZERO
                } else {
                    incoming
                };
                let facing = sample_normal.dot(-light_front);
                if facing < 0.0 {
                    radiance *= backface_hit_scale;
                }
                let facing_weight = facing.abs();
                let biased_theta = (theta - Vec2::splat(normal_angle)) * angle_bias;
                let sample_mask = angles_to_bitmask(biased_theta);
                let weight = (sample_mask & !slice_mask).count_ones() as f32 / 32.0;
                radiance *= facing_weight * weight;
                slice_sh.encode(light_front, radiance.extend(-weight));
                slice_mask |= sample_mask;
            }
        }
        accumulated = slice_sh.madd(projected_length, accumulated);
        total_length += projected_length;
        direction = Vec2::new(-direction.y, direction.x);
    }
    let sphere_weight = safe_rcp(total_length) * 2.0 * TAU;
    let gi_weight = sphere_weight / (1.5 - (angle_bias * (PI * 0.5)).cos());
    accumulated.scale_channels(Vec4::new(gi_weight, gi_weight, gi_weight, sphere_weight))
}

#[derive(Clone, Copy)]
struct ScreenSpaceRay {
    origin: Vec2,
    direction: Vec2,
    max_time: f32,
}

fn screen_space_ray(
    projection: Mat4,
    origin: Vec3,
    direction: Vec2,
    max_distance: f32,
    pixel_size: Vec2,
) -> ScreenSpaceRay {
    const BIAS: f32 = -2.4e-7 * 7.0;
    let mut start = projection * origin.extend(1.0);
    let mut end = projection * (origin + direction.extend(0.0) * max_distance).extend(1.0);
    start.z += BIAS;
    end.z += BIAS;
    let start_ndc = start.truncate() / start.w;
    let end_ndc = end.truncate() / end.w;
    let mut delta = end_ndc - start_ndc;
    if delta.truncate().length_squared() < 0.000_01 {
        delta.x = 0.000_01;
    }
    let ray_length_squared = delta.length_squared();
    let more_vertical = (delta.x / pixel_size.x).abs() < (delta.y / pixel_size.y).abs();
    let dominant = if more_vertical {
        delta.y.abs()
    } else {
        delta.x.abs()
    };
    delta /= dominant;
    delta *= if more_vertical {
        pixel_size.y
    } else {
        pixel_size.x
    };
    let mut max_time = (ray_length_squared * safe_rcp(delta.length_squared())).sqrt();
    max_time = max_time.min(line_unit_box_intersect_dist(start_ndc, delta));
    ScreenSpaceRay {
        origin: start_ndc.truncate() * 0.5 + Vec2::splat(0.5),
        direction: delta.truncate() * 0.5,
        max_time,
    }
}

fn line_unit_box_intersect_dist(origin: Vec3, direction: Vec3) -> f32 {
    let mut distance = f32::INFINITY;
    for axis in 0..3 {
        let safe_direction = if direction[axis] < 0.0 {
            -direction[axis].abs().max(1.0e-8)
        } else {
            direction[axis].abs().max(1.0e-8)
        };
        let first = (1.0 - origin[axis]) / safe_direction;
        let second = (-1.0 - origin[axis]) / safe_direction;
        distance = distance.min(first.max(second));
    }
    distance
}

#[derive(Clone, Copy)]
struct ScreenThicknessParameters {
    ndc_scale: f32,
    ndc_bias: f32,
    view_scale: f32,
    view_bias: f32,
}

impl ScreenThicknessParameters {
    fn pixel_depth_thickness_at(self, view_z: f32) -> f32 {
        let view_thickness = view_z * self.view_scale + self.view_bias;
        let ndc_thickness = view_thickness / (view_z * (view_z + view_thickness));
        ndc_thickness * self.ndc_scale + self.ndc_bias
    }
}

fn build_thickness_parameters(
    projection: Mat4,
    width: u32,
    height: u32,
    minimum_pixel_thickness: f32,
    minimum_constant_thickness: f32,
) -> ScreenThicknessParameters {
    let pixel_radius = Vec2::new(
        2.0 / projection.x_axis.x.abs() / width as f32,
        2.0 / projection.y_axis.y.abs() / height as f32,
    )
    .length()
        * minimum_pixel_thickness;
    if projection.w_axis.w == 0.0 {
        ScreenThicknessParameters {
            ndc_scale: projection.w_axis.z * 0.5,
            ndc_bias: 0.0,
            view_scale: pixel_radius,
            view_bias: -minimum_constant_thickness,
        }
    } else {
        ScreenThicknessParameters {
            ndc_scale: 0.0,
            ndc_bias: -(pixel_radius + minimum_constant_thickness) * projection.z_axis.z * 0.5,
            view_scale: 0.0,
            view_bias: 1.0,
        }
    }
}

struct ScreenThicknessEstimator {
    slope: Vec2,
    depth: f32,
    time: f32,
}

impl ScreenThicknessEstimator {
    fn new(start_depth: f32) -> Self {
        Self {
            slope: Vec2::ZERO,
            depth: start_depth,
            time: 0.0,
        }
    }

    fn thickness(&mut self, depth: f32, time: f32, minimum: f32) -> f32 {
        let slope = (depth - self.depth) / (time - self.time);
        let mut thickness = (depth - self.depth).abs();
        let change = (self.slope - Vec2::splat(slope)).abs();
        if change.x > self.slope.x.abs() || change.y > self.slope.y.abs() {
            thickness = 0.0;
        }
        thickness = thickness.max(minimum);
        self.slope.y = self.slope.x;
        self.slope.x = slope;
        self.time = time;
        self.depth = depth;
        thickness
    }
}

fn resolve(
    input: &FastGiInput<'_>,
    scale: u32,
    bias: [u32; 2],
    normal_mip: &SetupNormalMip,
    sh_image: &FastGiShTexture,
    closure_radiance: &mut [FastGiImage<[f32; 3]>; 3],
) -> Result<()> {
    let roughness_scale = 1.0 / 0.2;
    let roughness_bias = roughness_scale * input.settings.trace_max_roughness;
    for y in 0..input.height {
        for x in 0..input.width {
            let full_index = y as usize * input.width as usize + x as usize;
            if !input.processed[full_index] {
                continue;
            }
            let pixel = input.resolve[full_index];
            let has_fast_gi_closure = pixel.closures.iter().flatten().any(|closure| {
                (closure.roughness * roughness_scale - roughness_bias).clamp(0.0, 1.0) > 0.0
            });
            if !has_fast_gi_closure {
                continue;
            }
            let Some(probe) = pixel.volume_irradiance else {
                return Err(PotError::invalid_argument(
                    "FastGI resolve requires source volume-probe SH for each processed closure pixel",
                ));
            };
            let Some(mut sh) = resolve_sh(input, normal_mip, sh_image, x, y, scale, bias) else {
                continue;
            };
            sh = sh.rotate_view_to_world(input.view.view_inverse);
            let probe = probe.clamp_energy(input.settings.surface_indirect_clamp);
            for closure in pixel.closures.into_iter().flatten() {
                let factor = (closure.roughness * roughness_scale - roughness_bias).clamp(0.0, 1.0);
                if factor == 0.0 || closure.bin_index >= 3 {
                    continue;
                }
                let mut lighting = sh.evaluate_lambert(closure.dominant_direction);
                let mut distant_visibility = (lighting.w / 3.0).clamp(0.0, 1.0);
                if closure.transmission {
                    distant_visibility = 1.0;
                    lighting = Vec4::ZERO;
                }
                lighting += probe.evaluate_lambert(closure.dominant_direction) * distant_visibility;
                let output =
                    &mut closure_radiance[usize::from(closure.bin_index)].pixels[full_index];
                let input_radiance = Vec3::from_array(*output);
                let ray_radiance = if factor < 1.0 {
                    input_radiance
                } else {
                    Vec3::ZERO
                };
                let resolved = quantize_radiance(ray_radiance.lerp(lighting.truncate(), factor));
                *output = resolved.to_array();
            }
        }
    }
    Ok(())
}

fn resolve_sh(
    input: &FastGiInput<'_>,
    normal_mip: &SetupNormalMip,
    sh_image: &FastGiShTexture,
    full_x: u32,
    full_y: u32,
    scale: u32,
    bias: [u32; 2],
) -> Option<FastGiSh> {
    let low_x = full_x.saturating_sub(bias[0]) / scale;
    let low_y = full_y.saturating_sub(bias[1]) / scale;
    if scale == 1 {
        return sh_image.load(low_x, low_y);
    }
    let interpolation = Vec2::new(
        (full_x as f32 - (low_x * scale + bias[0]) as f32) / scale as f32,
        (full_y as f32 - (low_y * scale + bias[1]) as f32) / scale as f32,
    );
    let bilinear = [
        (1.0 - interpolation.x) * (1.0 - interpolation.y),
        interpolation.x * (1.0 - interpolation.y),
        (1.0 - interpolation.x) * interpolation.y,
        interpolation.x * interpolation.y,
    ];
    let offsets = [(0_u32, 0_u32), (1, 0), (0, 1), (1, 1)];
    let center_uv = (Vec2::new(full_x as f32, full_y as f32) + Vec2::splat(0.5))
        / Vec2::new(input.width as f32, input.height as f32);
    let center_depth = 1.0 - input.depth[(full_y * input.width + full_x) as usize];
    let center_position = point_screen_to_world(input.view, center_uv.extend(center_depth));
    let center_normal =
        Vec3::from_array(input.world_normals[(full_y * input.width + full_x) as usize]);
    let mut weights = [0.0_f32; 4];
    let mut samples = [FastGiSh::ZERO; 4];
    for index in 0..4 {
        let (dx, dy) = offsets[index];
        let sample_x = low_x + dx;
        let sample_y = low_y + dy;
        if sample_x >= normal_mip.width || sample_y >= normal_mip.height {
            continue;
        }
        let packed_normal = normal_mip.pixels[(sample_y * normal_mip.width + sample_x) as usize];
        let encoded_normal = unpack_normal(packed_normal);
        if encoded_normal[3] == 0.0 {
            continue;
        }
        let sample_full_x = sample_x * scale + bias[0];
        let sample_full_y = sample_y * scale + bias[1];
        if sample_full_x >= input.width || sample_full_y >= input.height {
            continue;
        }
        let sample_depth =
            1.0 - input.depth[(sample_full_y * input.width + sample_full_x) as usize];
        let sample_uv = (Vec2::new(sample_full_x as f32, sample_full_y as f32) + Vec2::splat(0.5))
            / Vec2::new(input.width as f32, input.height as f32);
        let sample_position = point_screen_to_world(input.view, sample_uv.extend(sample_depth));
        let sample_normal = view_normal_to_world(
            input.view,
            Vec3::new(encoded_normal[0], encoded_normal[1], encoded_normal[2]) * 2.0 - Vec3::ONE,
        );
        let depth_weight = planar_weight(center_normal, center_position, sample_position, 10_000.0);
        let normal_weight = angle_weight(center_normal, sample_normal);
        weights[index] = (depth_weight * normal_weight).max(1.0e-4) * bilinear[index];
        if weights[index] > 0.0 {
            samples[index] = sh_image.load(sample_x, sample_y)?;
        }
    }
    let total = weights.iter().sum::<f32>();
    let inverse_total = safe_rcp(total);
    for weight in &mut weights {
        *weight *= inverse_total;
    }
    let mut result = samples[0].scale(weights[0]);
    for index in 1..4 {
        result = samples[index].madd(weights[index], result);
    }
    Some(result)
}

fn sample_radiance_mip(mips: &[SetupRadianceMip; 4], uv: Vec2, lod: f32) -> Vec3 {
    let lod = lod.clamp(0.0, 3.0);
    let lower = lod.floor() as usize;
    let upper = (lower + 1).min(3);
    let blend = lod - lower as f32;
    sample_radiance_level(&mips[lower], uv).lerp(sample_radiance_level(&mips[upper], uv), blend)
}

fn sample_radiance_level(mip: &SetupRadianceMip, uv: Vec2) -> Vec3 {
    sample_linear_packed(
        uv,
        mip.width,
        mip.height,
        &mip.pixels,
        unpack_ufloat_r11g11b10,
    )
}

fn sample_normal_mip(mips: &[SetupNormalMip; 4], uv: Vec2, lod: f32) -> Vec3 {
    let lod = lod.clamp(0.0, 3.0);
    let lower = lod.floor() as usize;
    let upper = (lower + 1).min(3);
    let blend = lod - lower as f32;
    sample_normal_level(&mips[lower], uv).lerp(sample_normal_level(&mips[upper], uv), blend)
}

fn sample_normal_level(mip: &SetupNormalMip, uv: Vec2) -> Vec3 {
    sample_linear_packed(uv, mip.width, mip.height, &mip.pixels, |bits| {
        let value = unpack_normal(bits);
        Vec3::new(value[0], value[1], value[2])
    }) * 2.0
        - Vec3::ONE
}

fn sample_linear_packed<F>(uv: Vec2, width: u32, height: u32, pixels: &[u32], mut decode: F) -> Vec3
where
    F: FnMut(u32) -> Vec3,
{
    let position = uv * Vec2::new(width as f32, height as f32) - Vec2::splat(0.5);
    let base = position.floor().as_ivec2();
    let fraction = position.fract();
    let sample = |x: i32, y: i32, decode: &mut F| {
        let x = x.clamp(0, width as i32 - 1) as u32;
        let y = y.clamp(0, height as i32 - 1) as u32;
        decode(pixels[y as usize * width as usize + x as usize])
    };
    let top = sample(base.x, base.y, &mut decode)
        .lerp(sample(base.x + 1, base.y, &mut decode), fraction.x);
    let bottom = sample(base.x, base.y + 1, &mut decode)
        .lerp(sample(base.x + 1, base.y + 1, &mut decode), fraction.x);
    top.lerp(bottom, fraction.y)
}

fn sample_hiz(mips: &[FastGiTexture<'_, f32>], uv: Vec2, lod: i32) -> f32 {
    let mip = mips[lod.clamp(0, mips.len() as i32 - 1) as usize];
    let x = (uv.x * mip.width as f32).floor() as i32;
    let y = (uv.y * mip.height as f32).floor() as i32;
    let x = x.clamp(0, mip.width as i32 - 1) as u32;
    let y = y.clamp(0, mip.height as i32 - 1) as u32;
    mip.pixels[y as usize * mip.width as usize + x as usize]
}

fn hiz_fetch(mips: &[FastGiTexture<'_, f32>], x: u32, y: u32, lod: usize) -> f32 {
    let mip = mips[lod.min(mips.len() - 1)];
    let x = x.min(mip.width - 1);
    let y = y.min(mip.height - 1);
    mip.pixels[y as usize * mip.width as usize + x as usize]
}

fn sample_nearest<T: Copy>(texture: FastGiTexture<'_, T>, pixel: Vec2) -> T {
    let x = (pixel.x.floor() as i32).rem_euclid(texture.width as i32) as u32;
    let y = (pixel.y.floor() as i32).rem_euclid(texture.height as i32) as u32;
    texture.pixels[y as usize * texture.width as usize + x as usize]
}

fn sample_linear_rgba16(texture: FastGiTexture<'_, [f32; 4]>, uv: Vec2) -> Vec4 {
    let position = uv * Vec2::new(texture.width as f32, texture.height as f32) - Vec2::splat(0.5);
    let base = position.floor().as_ivec2();
    let fraction = position.fract();
    let sample = |x: i32, y: i32| {
        let x = x.clamp(0, texture.width as i32 - 1) as u32;
        let y = y.clamp(0, texture.height as i32 - 1) as u32;
        Vec4::from_array(
            texture.pixels[y as usize * texture.width as usize + x as usize]
                .map(|value| f16_to_f32(f32_to_f16(value))),
        )
    };
    let top = sample(base.x, base.y).lerp(sample(base.x + 1, base.y), fraction.x);
    let bottom = sample(base.x, base.y + 1).lerp(sample(base.x + 1, base.y + 1), fraction.x);
    top.lerp(bottom, fraction.y)
}

fn copy_radiance(texture: FastGiTexture<'_, [f32; 3]>) -> Result<FastGiImage<[f32; 3]>> {
    let mut pixels = allocate(texture.pixels.len(), [0.0; 3], "FastGI closure radiance")?;
    for (destination, source) in pixels.iter_mut().zip(texture.pixels) {
        *destination = quantize_radiance(Vec3::from_array(*source)).to_array();
    }
    Ok(FastGiImage { pixels })
}

fn point_screen_to_view(view: FastGiView, screen: Vec3) -> Vec3 {
    let ndc = screen * 2.0 - Vec3::ONE;
    let homogeneous = view.projection_inverse * ndc.extend(1.0);
    homogeneous.truncate() / homogeneous.w
}

fn point_screen_to_world(view: FastGiView, screen: Vec3) -> Vec3 {
    view.view_inverse
        .transform_point3(point_screen_to_view(view, screen))
}

fn normal_world_to_view(view: FastGiView, normal: Vec3) -> Vec3 {
    view.view.transform_vector3(normal)
}

fn view_normal_to_world(view: FastGiView, normal: Vec3) -> Vec3 {
    view.view_inverse.transform_vector3(normal)
}

fn angles_to_bitmask(theta: Vec2) -> u32 {
    let ratio = (theta / PI + Vec2::splat(0.5)).clamp(Vec2::ZERO, Vec2::ONE);
    let start = (32.0 * ratio.x).floor() as u32;
    let count = (32.0 * (ratio.y - ratio.x)).ceil() as u32;
    let bits = if count < 32 {
        (1_u32 << count) - 1
    } else {
        u32::MAX
    };
    if start >= 32 { 0 } else { bits << start }
}

fn sample_circle(value: f32) -> Vec2 {
    let angle = (value - 0.5) * TAU;
    let cosine = angle.cos();
    let sine = (1.0 - cosine * cosine).max(0.0).sqrt() * sign(angle);
    Vec2::new(cosine, sine)
}

fn safe_normalize_and_length(value: Vec3, length: &mut f32) -> Vec3 {
    let squared_length = value.length_squared();
    if squared_length > 1.0e-35 {
        *length = squared_length.sqrt();
        value / *length
    } else {
        *length = 0.0;
        Vec3::X
    }
}

fn safe_rcp(value: f32) -> f32 {
    if value.partial_cmp(&0.0) == Some(std::cmp::Ordering::Equal) {
        0.0
    } else {
        1.0 / value
    }
}

fn sign(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn fract(value: f32) -> f32 {
    value - value.floor()
}

fn acos_fast(value: f32) -> f32 {
    let absolute = value.abs();
    let root = sqrt_fast(1.0 - absolute);
    let result = (-0.156_583 * absolute + PI * 0.5) * root;
    if value >= 0.0 { result } else { PI - result }
}

fn sqrt_fast(value: f32) -> f32 {
    f32::from_bits(0x1fbd_1df5_u32.wrapping_add((value.to_bits() as i32 >> 1) as u32))
}

fn gaussian_factor(linear_distance: f32, standard_deviation: f32) -> f32 {
    std::f32::consts::LOG2_E * standard_deviation / (linear_distance * linear_distance)
}

fn gaussian_weight(factor: f32, square_distance: f32) -> f32 {
    (-factor * square_distance).exp2()
}

fn planar_weight(plane_normal: Vec3, plane_position: Vec3, position: Vec3, scale: f32) -> f32 {
    let distance = plane_normal.dot(position - plane_position);
    gaussian_weight(scale, distance * distance)
}

fn angle_weight(center: Vec3, sample: Vec3) -> f32 {
    center.dot(sample).powi(8).clamp(0.0, 1.0)
}

fn sh_compress(sh: FastGiSh) -> FastGiSh {
    let factor = Vec4::new(
        safe_rcp(sh.l0_m0.x * SH_SQRT3),
        safe_rcp(sh.l0_m0.y * SH_SQRT3),
        safe_rcp(sh.l0_m0.z * SH_SQRT3),
        safe_rcp(sh.l0_m0.w * SH_SQRT3),
    );
    FastGiSh {
        l0_m0: sh.l0_m0,
        l1_mn1: sh.l1_mn1 * factor * 0.5 + Vec4::splat(0.5),
        l1_m0: sh.l1_m0 * factor * 0.5 + Vec4::splat(0.5),
        l1_mp1: sh.l1_mp1 * factor * 0.5 + Vec4::splat(0.5),
    }
}

fn unpack_normal(bits: u32) -> [f32; 4] {
    [
        (bits & 1023) as f32 / 1023.0,
        ((bits >> 10) & 1023) as f32 / 1023.0,
        ((bits >> 20) & 1023) as f32 / 1023.0,
        (bits >> 30) as f32 / 3.0,
    ]
}

fn pack_unorm1010102(value: Vec3, alpha: u32) -> u32 {
    unorm_bits(value.x, 1023)
        | (unorm_bits(value.y, 1023) << 10)
        | (unorm_bits(value.z, 1023) << 20)
        | ((alpha & 3) << 30)
}

fn unorm8(value: f32) -> u8 {
    unorm_bits(value, 255) as u8
}

fn unorm_bits(value: f32, maximum: u32) -> u32 {
    let scaled = value.clamp(0.0, 1.0) * maximum as f32;
    let floor = scaled.floor();
    let integer = floor as u32;
    let fraction = scaled - floor;
    let round_up = match fraction.partial_cmp(&0.5) {
        Some(std::cmp::Ordering::Greater) => true,
        Some(std::cmp::Ordering::Equal) => integer & 1 != 0,
        _ => false,
    };
    integer + u32::from(round_up)
}

fn pack_ufloat_r11g11b10(rgb: Vec3) -> u32 {
    f32_to_ufloat(rgb.x, 6) | (f32_to_ufloat(rgb.y, 6) << 11) | (f32_to_ufloat(rgb.z, 5) << 22)
}

fn unpack_ufloat_r11g11b10(packed: u32) -> Vec3 {
    Vec3::new(
        ufloat_to_f32(packed & 0x7ff, 6),
        ufloat_to_f32((packed >> 11) & 0x7ff, 6),
        ufloat_to_f32((packed >> 22) & 0x3ff, 5),
    )
}

fn quantize_radiance(value: Vec3) -> Vec3 {
    unpack_ufloat_r11g11b10(pack_ufloat_r11g11b10(value))
}

fn f32_to_ufloat(value: f32, mantissa_bits: u32) -> u32 {
    if value.is_nan() {
        return (31 << mantissa_bits) | (1 << (mantissa_bits - 1));
    }
    if value <= 0.0 {
        return 0;
    }
    if value.is_infinite() {
        return 31 << mantissa_bits;
    }
    let bits = value.to_bits();
    let exponent_field = (bits >> 23) & 0xff;
    let (exponent, significand) = if exponent_field == 0 {
        (-126, bits & 0x7f_ff_ff)
    } else {
        (
            exponent_field as i32 - 127,
            (bits & 0x7f_ff_ff) | 0x80_00_00,
        )
    };
    let encoded_exponent = exponent + 15;
    if encoded_exponent <= 0 {
        let shift = (9 - mantissa_bits as i32 - exponent).max(0) as u32;
        return round_shift_even(significand, shift).min(1 << mantissa_bits);
    }
    if encoded_exponent >= 31 {
        return 31 << mantissa_bits;
    }
    let mut rounded = round_shift_even(significand, 23 - mantissa_bits);
    let mut output_exponent = encoded_exponent as u32;
    if rounded >= 1 << (mantissa_bits + 1) {
        rounded >>= 1;
        output_exponent += 1;
    }
    if output_exponent >= 31 {
        return 31 << mantissa_bits;
    }
    (output_exponent << mantissa_bits) | (rounded - (1 << mantissa_bits))
}

fn ufloat_to_f32(bits: u32, mantissa_bits: u32) -> f32 {
    let mantissa = bits & ((1 << mantissa_bits) - 1);
    let exponent = (bits >> mantissa_bits) & 31;
    if exponent == 0 {
        return mantissa as f32 * 2.0_f32.powi(-14 - mantissa_bits as i32);
    }
    if exponent == 31 {
        return if mantissa == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        };
    }
    (1.0 + mantissa as f32 / (1 << mantissa_bits) as f32) * 2.0_f32.powi(exponent as i32 - 15)
}

fn round_shift_even(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    if shift > 31 {
        return 0;
    }
    let truncated = value >> shift;
    let remainder = value & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    truncated + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
}

fn f32_to_f16(value: f32) -> u16 {
    let sign = (value.to_bits() >> 16) & 0x8000;
    (sign | f32_to_ufloat(value.abs(), 10)) as u16
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = if value & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = (value >> 10) & 31;
    let mantissa = value & 1023;
    if exponent == 0 {
        return sign * f32::from(mantissa) * 2.0_f32.powi(-24);
    }
    if exponent == 31 {
        return if mantissa == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    sign * (1.0 + f32::from(mantissa) / 1024.0) * 2.0_f32.powi(i32::from(exponent) - 15)
}

fn checked_pixel_count(width: u32, height: u32) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(PotError::invalid_argument(
            "FastGI dimensions must be non-zero",
        ));
    }
    usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| PotError::invalid_argument("FastGI dimensions exceed addressable memory"))
}

fn validate_texture<T>(texture: &FastGiTexture<'_, T>) -> Result<()> {
    if checked_pixel_count(texture.width, texture.height)? != texture.pixels.len() {
        return Err(PotError::invalid_argument(
            "FastGI texture dimensions do not match its pixel data",
        ));
    }
    Ok(())
}

fn allocate<T: Clone>(length: usize, value: T, label: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| memory_error(label))?;
    values.resize(length, value);
    Ok(values)
}

fn memory_error(label: &str) -> PotError {
    PotError::new(
        ErrorCode::LimitExceeded,
        format!("{label} exceeds available memory"),
    )
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

fn ceil_to_multiple(value: u32, multiple: u32) -> u32 {
    div_ceil(value, multiple) * multiple
}

fn compact_bits(mut value: u32) -> u32 {
    value &= 0x5555_5555;
    value = (value ^ (value >> 1)) & 0x3333_3333;
    value = (value ^ (value >> 2)) & 0x0f0f_0f0f;
    value = (value ^ (value >> 4)) & 0x00ff_00ff;
    value = (value ^ (value >> 8)) & 0x0000_ffff;
    value
}

fn random_in_tile(sample_index: u32, tile_size: u32) -> [u32; 2] {
    if tile_size == 1 {
        return [0, 0];
    }
    let mut hash = sample_index;
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x85eb_ca6b);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0xc2b2_ae35);
    hash ^= hash >> 16;
    let code = hash % (tile_size * tile_size);
    [compact_bits(code), compact_bits(code >> 1)]
}
