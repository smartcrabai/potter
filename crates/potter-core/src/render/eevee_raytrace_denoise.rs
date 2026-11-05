use crate::error::{ErrorCode, PotError, Result};

pub const RAYTRACE_GROUP_SIZE: u32 = 8;
pub const INVALID_RADIANCE: [f32; 3] = [65_024.0, 65_024.0, 64_512.0];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    pub width: u32,
    pub height: u32,
}

impl Extent {
    fn pixel_count(self) -> Result<usize> {
        if self.width == 0 || self.height == 0 {
            return Err(PotError::invalid_argument(
                "denoising image dimensions must be non-zero",
            ));
        }
        if self.width > i32::MAX as u32 || self.height > i32::MAX as u32 {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "denoising dimensions exceed signed shader coordinate limits",
            ));
        }
        usize::try_from(self.width)
            .ok()
            .and_then(|width| {
                usize::try_from(self.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "denoising image dimensions exceed platform limits",
                )
            })
    }

    fn tile_extent(self) -> Result<Self> {
        self.pixel_count()?;
        Ok(Self {
            width: self.width.div_ceil(RAYTRACE_GROUP_SIZE),
            height: self.height.div_ceil(RAYTRACE_GROUP_SIZE),
        })
    }

    fn index(self, x: u32, y: u32) -> Option<usize> {
        if x >= self.width || y >= self.height {
            return None;
        }
        usize::try_from(y)
            .ok()?
            .checked_mul(usize::try_from(self.width).ok()?)?
            .checked_add(usize::try_from(x).ok()?)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RequestedStages {
    pub spatial: bool,
    pub temporal: bool,
    pub bilateral: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagePlan {
    /// The spatial dispatch always runs; false selects its source upsample-only branch.
    pub spatial_denoising: bool,
    pub temporal: bool,
    pub bilateral: bool,
}

/// Matches `RayTraceModule::trace`'s dependency gates; spatial dispatch still runs in upsample-only mode.
pub const fn resolve_stage_plan(denoise_enabled: bool, requested: RequestedStages) -> StagePlan {
    let spatial_denoising = denoise_enabled && requested.spatial;
    let temporal = spatial_denoising && requested.temporal;
    let bilateral = temporal && requested.bilateral;
    StagePlan {
        spatial_denoising,
        temporal,
        bilateral,
    }
}

/// Decoded `SFLOAT_16_16_16_16` ray direction/PDF-inverse input, `SFLOAT_32` ray time, and scene-linear `UFLOAT_11_11_10` radiance.
#[derive(Debug, Clone, Copy)]
pub struct RaySample {
    pub direction_and_pdf_inverse: [f32; 4],
    pub time: f32,
    pub radiance: [f32; 3],
}

/// Per-full-resolution-pixel G-buffer and view values, including unbiased
/// view-space position deltas; the spatial stage applies their roughness bias.
#[derive(Debug, Clone, Copy)]
pub struct SpatialPixel {
    pub closure_present: bool,
    pub apparent_roughness: f32,
    /// Final `fract(blue_noise + closure_rng)` value from `spatial_main`.
    pub noise: [f32; 2],
    pub view_normal: [f32; 3],
    pub view_incident: [f32; 3],
    pub view_position_derivatives: [[f32; 3]; 2],
    pub scene_view_z: f32,
}

#[derive(Clone, Copy)]
pub struct SpatialInput<'a> {
    pub extent: Extent,
    /// Trace image extent. Must cover the source ceil-divided trace extent; extra border storage is allowed.
    pub trace_extent: Extent,
    pub trace_scale: u32,
    pub trace_offset: [u32; 2],
    /// One closure's `RAYTRACE_TILEMASK_FORMAT` values; bit zero marks a scheduled tile.
    pub tile_mask: &'a [u8],
    pub pixels: &'a [SpatialPixel],
    pub rays: &'a [RaySample],
    /// True selects the source's `skip_denoise` branches.
    pub skip_denoise: bool,
}

pub struct SpatialOutput<'a> {
    /// Decoded values quantized to `UFLOAT_11_11_10` at each source image write.
    pub radiance: &'a mut [[f32; 3]],
    /// Decoded `SFLOAT_16` values quantized at each source image write.
    pub variance: &'a mut [f32],
    /// `SFLOAT_32` hit depth.
    pub hit_depth: &'a mut [f32],
}

/// EEVEE closure, view, and upsample weights consumed by the spatial shader.
pub trait SpatialKernelOps: ColorTransfer {
    fn closure_evaluate_pdf(
        &mut self,
        pixel_index: usize,
        ray_direction: [f32; 3],
        view_normal: [f32; 3],
        view_incident: [f32; 3],
    ) -> f32;
    fn depth_view_to_screen(&mut self, pixel_index: usize, view_z: f32) -> f32;
    fn upsample_sample_weight(&mut self, center_fullres: [u32; 2], trace_sample: [i32; 2]) -> f32;
}

/// EEVEE scene-linear/log conversion and `safe_color` operations used verbatim by the shader.
pub trait ColorTransfer {
    fn log_from_scene_linear(&self, color: [f32; 3]) -> [f32; 3];
    fn scene_linear_from_log(&self, color: [f32; 3]) -> [f32; 3];
    fn ycocg_from_scene_linear(&self, color: [f32; 3]) -> [f32; 3];
    fn scene_linear_from_ycocg(&self, color: [f32; 3]) -> [f32; 3];
    fn safe_color(&self, color: [f32; 3]) -> [f32; 3];
}

/// Executes the spatial shader over active 8x8 tiles, including skip-denoise bilateral upsampling.
pub fn spatial_denoise(
    input: SpatialInput<'_>,
    output: SpatialOutput<'_>,
    ops: &mut impl SpatialKernelOps,
) -> Result<()> {
    let SpatialOutput {
        radiance,
        variance,
        hit_depth,
    } = output;
    let pixel_count = input.extent.pixel_count()?;
    let tile_extent = input.extent.tile_extent()?;
    let tile_count = tile_extent.pixel_count()?;
    if input.trace_scale == 0
        || input.trace_offset[0] >= input.trace_scale
        || input.trace_offset[1] >= input.trace_scale
    {
        return Err(PotError::invalid_argument(
            "spatial trace scale or offset is invalid",
        ));
    }
    let expected_trace = Extent {
        width: input.extent.width.div_ceil(input.trace_scale),
        height: input.extent.height.div_ceil(input.trace_scale),
    };
    if input.trace_extent.width < expected_trace.width
        || input.trace_extent.height < expected_trace.height
    {
        return Err(PotError::invalid_argument(
            "spatial trace extent is smaller than the ceil-divided render extent",
        ));
    }
    check_len("spatial tile mask", input.tile_mask.len(), tile_count)?;
    check_len("spatial pixel inputs", input.pixels.len(), pixel_count)?;
    check_len(
        "spatial ray samples",
        input.rays.len(),
        input.trace_extent.pixel_count()?,
    )?;
    check_len("spatial radiance output", radiance.len(), pixel_count)?;
    check_len("spatial variance output", variance.len(), pixel_count)?;
    check_len("spatial hit-depth output", hit_depth.len(), pixel_count)?;

    for tile_y in 0..tile_extent.height {
        for tile_x in 0..tile_extent.width {
            let tile_index = tile_extent.index(tile_x, tile_y).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "spatial tile index overflow")
            })?;
            if input.tile_mask[tile_index] & 1 == 0 {
                continue;
            }
            if !input.skip_denoise {
                clear_spatial_neighbor_tiles(
                    input.extent,
                    tile_extent,
                    input.tile_mask,
                    [tile_x, tile_y],
                    radiance,
                    variance,
                    hit_depth,
                );
            }
            let x_end = ((tile_x + 1) * RAYTRACE_GROUP_SIZE).min(input.extent.width);
            let y_end = ((tile_y + 1) * RAYTRACE_GROUP_SIZE).min(input.extent.height);
            for y in tile_y * RAYTRACE_GROUP_SIZE..y_end {
                for x in tile_x * RAYTRACE_GROUP_SIZE..x_end {
                    let index = input.extent.index(x, y).ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "spatial pixel index overflow")
                    })?;
                    if input.skip_denoise {
                        if input.trace_scale == 1 {
                            let Some(ray_index) = input.trace_extent.index(x, y) else {
                                continue;
                            };
                            let linear_pre_store = input.rays[ray_index].radiance;
                            let stored_radiance = quantize_radiance(linear_pre_store);
                            radiance[index] = stored_radiance;
                        } else {
                            let shifted = trace_coordinate([x, y], input.trace_offset);
                            let nearest = [
                                shifted[0] / input.trace_scale as i32,
                                shifted[1] / input.trace_scale as i32,
                            ];
                            let subpixel = [
                                fract(shifted[0] as f32 / input.trace_scale as f32),
                                fract(shifted[1] as f32 / input.trace_scale as f32),
                            ];
                            let neighbors = [
                                [nearest[0], nearest[1] + 1],
                                [nearest[0] + 1, nearest[1] + 1],
                                [nearest[0] + 1, nearest[1]],
                                [nearest[0], nearest[1]],
                            ];
                            let bilinear = bilinear_weights(subpixel);
                            let mut accumulated = [0.0; 3];
                            let mut weight_sum = 0.0;
                            for sample in 0..4 {
                                let coordinate = neighbors[sample];
                                let Some(ray_index) = trace_index(input.trace_extent, coordinate)
                                else {
                                    continue;
                                };
                                let ray = input.rays[ray_index];
                                if ray.direction_and_pdf_inverse[3] == 0.0 {
                                    continue;
                                }
                                let bilateral = ops.upsample_sample_weight([x, y], coordinate);
                                let weight = bilinear[sample] * bilateral;
                                let log_radiance = ops.log_from_scene_linear(ray.radiance);
                                for (accumulated_channel, log_channel) in
                                    accumulated.iter_mut().zip(log_radiance)
                                {
                                    *accumulated_channel += log_channel * weight;
                                }
                                weight_sum += weight;
                            }
                            for accumulated_channel in &mut accumulated {
                                *accumulated_channel *= safe_rcp(weight_sum);
                            }
                            let linear_pre_store = ops.scene_linear_from_log(accumulated);
                            let stored_radiance = quantize_radiance(linear_pre_store);
                            radiance[index] = stored_radiance;
                        }
                        continue;
                    }

                    let pixel = input.pixels[index];
                    if !pixel.closure_present {
                        write_invalid(index, radiance, variance, hit_depth);
                        continue;
                    }
                    let shifted = trace_coordinate([x, y], input.trace_offset);
                    let nearest = [
                        shifted[0] / input.trace_scale as i32,
                        shifted[1] / input.trace_scale as i32,
                    ];
                    let bilinear_co = [
                        fract(shifted[0] as f32 / input.trace_scale as f32),
                        fract(shifted[1] as f32 / input.trace_scale as f32),
                    ];
                    let center_sample = [
                        nearest[0] + i32::from(bilinear_co[0] > pixel.noise[0]),
                        nearest[1] + i32::from(bilinear_co[1] > pixel.noise[1]),
                    ];
                    let roughness = pixel.apparent_roughness;
                    let filter_size_factor = saturate(roughness * 8.0);
                    let mut sample_count = 1 + (15.0 * filter_size_factor + 0.5).floor() as u32;
                    let filter_radius = 8.0 * filter_size_factor.sqrt();
                    let min_filter_radius = if input.trace_scale > 1 { 1.5 } else { 0.0 };
                    if input.trace_scale > 1 {
                        sample_count = sample_count.max(5);
                    }
                    let max_filter_radius = (RAYTRACE_GROUP_SIZE - 1) as f32;
                    let bias = 0.2_f32.max(saturate((0.75 - roughness) / (0.75 - 0.3)));
                    let dpdx = scale3(pixel.view_position_derivatives[0], bias);
                    let dpdy = scale3(pixel.view_position_derivatives[1], bias);
                    let filter_up = normalize2([pixel.view_normal[0], pixel.view_normal[1]]);
                    let filter_right = [-filter_up[1], filter_up[0]];
                    let aspect =
                        1.0 - 0.5 * saturate((roughness * 12.0).min(2.0 - roughness * 5.0));
                    let radius_x = filter_radius.clamp(min_filter_radius, max_filter_radius);
                    let radius_y =
                        (filter_radius * aspect).clamp(min_filter_radius, max_filter_radius);
                    let rotation = [
                        [filter_up[0] * radius_x, filter_right[0] * radius_y],
                        [filter_up[1] * radius_x, filter_right[1] * radius_y],
                    ];
                    let mut radiance_accum = [0.0; 3];
                    let mut moment = [0.0; 3];
                    let mut weight_accum = 0.0;
                    let mut closest_hit_time = 1.0e10_f32;
                    for sample in 0..sample_count {
                        let mut xi = hammersley_2d(sample, sample_count);
                        xi[1] *= 1.0 - pixel.noise[0] / sample_count as f32;
                        let disk = sample_disk(xi);
                        let offset_float = [
                            rotation[0][0] * disk[0] + rotation[0][1] * disk[1],
                            rotation[1][0] * disk[0] + rotation[1][1] * disk[1],
                        ];
                        let offset = [
                            (offset_float[0] + 0.5).floor() as i32,
                            (offset_float[1] + 0.5).floor() as i32,
                        ];
                        let sample_texel =
                            [center_sample[0] + offset[0], center_sample[1] + offset[1]];
                        let Some(ray_index) = trace_index(input.trace_extent, sample_texel) else {
                            continue;
                        };
                        let ray = input.rays[ray_index];
                        let ray_pdf_inverse = ray.direction_and_pdf_inverse[3].abs();
                        if ray_pdf_inverse == 0.0 {
                            continue;
                        }
                        closest_hit_time = closest_hit_time.min(ray.time);
                        let ray_direction = [
                            ray.direction_and_pdf_inverse[0],
                            ray.direction_and_pdf_inverse[1],
                            ray.direction_and_pdf_inverse[2],
                        ];
                        let corrected_direction = normalize3(add3(
                            add3(
                                scale3(dpdx, offset[0] as f32),
                                scale3(dpdy, offset[1] as f32),
                            ),
                            scale3(ray_direction, ray.time),
                        ));
                        let pdf = ops.closure_evaluate_pdf(
                            index,
                            corrected_direction,
                            pixel.view_normal,
                            pixel.view_incident,
                        );
                        let weight = (pdf * ray_pdf_inverse).min(1.0e30);
                        let log_radiance = ops.log_from_scene_linear(ray.radiance);
                        for channel in 0..3 {
                            radiance_accum[channel] += log_radiance[channel] * weight;
                            moment[channel] +=
                                log_radiance[channel] * log_radiance[channel] * weight;
                        }
                        weight_accum += weight;
                    }
                    let inv_weight = safe_rcp(weight_accum);
                    for channel in 0..3 {
                        radiance_accum[channel] *= inv_weight;
                        moment[channel] *= inv_weight;
                    }
                    let mut hit_variance = 0.0_f32;
                    for channel in 0..3 {
                        let variance = (moment[channel]
                            - radiance_accum[channel] * radiance_accum[channel])
                            .abs()
                            * safe_rcp(radiance_accum[channel]);
                        hit_variance = hit_variance.max(variance);
                    }
                    let hit_depth_value =
                        ops.depth_view_to_screen(index, pixel.scene_view_z - closest_hit_time);
                    let linear_pre_store = ops.scene_linear_from_log(radiance_accum);
                    let stored_radiance = quantize_radiance(linear_pre_store);
                    radiance[index] = stored_radiance;
                    variance[index] = quantize_f16(hit_variance);
                    hit_depth[index] = hit_depth_value;
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct TemporalPixel {
    /// World position reconstructed from the current reverse-Z surface depth.
    pub surface_world: [f32; 3],
    /// World position reconstructed from the spatial pass's `SFLOAT_32` hit depth.
    pub hit_world: [f32; 3],
}

#[derive(Clone, Copy)]
pub struct TemporalInput<'a> {
    pub extent: Extent,
    /// One closure's `RAYTRACE_TILEMASK_FORMAT` values; bit zero marks a scheduled tile.
    pub tile_mask: &'a [u8],
    /// Decoded `UFLOAT_11_11_10` spatial radiance.
    pub spatial_radiance: &'a [[f32; 3]],
    /// Decoded `SFLOAT_16` spatial variance.
    pub spatial_variance: &'a [f32],
    pub pixels: &'a [TemporalPixel],
    /// Required prior-frame buffers; values are decoded from the source UFLOAT/SFLOAT formats.
    pub history_extent: Extent,
    pub history_radiance: &'a [[f32; 3]],
    pub history_variance: &'a [f32],
    pub history_tile_mask: &'a [u8],
    /// World-to-previous-clip matrix in row-major order.
    pub history_view_projection: [[f32; 4]; 4],
}

pub struct TemporalOutput<'a> {
    pub radiance: &'a mut [[f32; 3]],
    pub variance: &'a mut [f32],
}

/// Reprojects and accumulates the spatial result using caller-provided prior-frame buffers.
pub fn temporal_denoise(
    input: TemporalInput<'_>,
    output: TemporalOutput<'_>,
    color: &impl ColorTransfer,
) -> Result<()> {
    let TemporalOutput { radiance, variance } = output;
    let pixel_count = input.extent.pixel_count()?;
    let tile_extent = input.extent.tile_extent()?;
    let tile_count = tile_extent.pixel_count()?;
    let history_count = input.history_extent.pixel_count()?;
    let history_tiles = input.history_extent.tile_extent()?;
    let history_tile_count = history_tiles.pixel_count()?;
    check_len("temporal tile mask", input.tile_mask.len(), tile_count)?;
    check_len(
        "temporal radiance",
        input.spatial_radiance.len(),
        pixel_count,
    )?;
    check_len(
        "temporal variance",
        input.spatial_variance.len(),
        pixel_count,
    )?;
    check_len("temporal pixels", input.pixels.len(), pixel_count)?;
    check_len(
        "history radiance",
        input.history_radiance.len(),
        history_count,
    )?;
    check_len(
        "history variance",
        input.history_variance.len(),
        history_count,
    )?;
    check_len(
        "history tile mask",
        input.history_tile_mask.len(),
        history_tile_count,
    )?;
    check_len("temporal radiance output", radiance.len(), pixel_count)?;
    check_len("temporal variance output", variance.len(), pixel_count)?;

    for tile_y in 0..tile_extent.height {
        for tile_x in 0..tile_extent.width {
            let tile_index = tile_extent.index(tile_x, tile_y).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "temporal tile index overflow")
            })?;
            if input.tile_mask[tile_index] & 1 == 0 {
                continue;
            }
            clear_temporal_neighbor_variance(
                input.extent,
                tile_extent,
                input.tile_mask,
                [tile_x, tile_y],
                variance,
            );
            let x_end = ((tile_x + 1) * RAYTRACE_GROUP_SIZE).min(input.extent.width);
            let y_end = ((tile_y + 1) * RAYTRACE_GROUP_SIZE).min(input.extent.height);
            for y in tile_y * RAYTRACE_GROUP_SIZE..y_end {
                for x in tile_x * RAYTRACE_GROUP_SIZE..x_end {
                    let index = input.extent.index(x, y).ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "temporal pixel index overflow")
                    })?;
                    let in_radiance = input.spatial_radiance[index];
                    let in_variance = input.spatial_variance[index];
                    if is_invalid_radiance(in_radiance) {
                        radiance[index] = quantize_radiance(INVALID_RADIANCE);
                        variance[index] = quantize_f16(0.0);
                        continue;
                    }
                    let local = local_statistics(
                        input.extent,
                        input.spatial_radiance,
                        [x, y],
                        in_radiance,
                        color,
                    );
                    let pixel = input.pixels[index];
                    let mut history_radiance = sample_history_radiance(
                        input.history_extent,
                        input.history_radiance,
                        input.history_tile_mask,
                        history_tiles,
                        input.history_view_projection,
                        pixel.surface_world,
                        local,
                        color,
                    );
                    let reflected = sample_history_radiance(
                        input.history_extent,
                        input.history_radiance,
                        input.history_tile_mask,
                        history_tiles,
                        input.history_view_projection,
                        pixel.hit_world,
                        local,
                        color,
                    );
                    add4(&mut history_radiance, reflected);
                    let reciprocal = safe_rcp(history_radiance[3]);
                    let mut history_ycocg = [0.0; 3];
                    for channel in 0..3 {
                        history_ycocg[channel] = (history_radiance[channel] * reciprocal)
                            .clamp(local.clamp_min[channel], local.clamp_max[channel]);
                    }
                    let history_color = color.scene_linear_from_ycocg(history_ycocg);
                    let blend = if history_radiance[3] * reciprocal > 1.0e-3 {
                        0.97 * (0.75 + 0.25 * saturate(in_variance * 20.0))
                    } else {
                        0.0
                    };
                    let out_radiance = mix3(
                        color.safe_color(in_radiance),
                        color.safe_color(history_color),
                        blend,
                    );
                    radiance[index] = quantize_radiance(out_radiance);
                    let history_variance = sample_history_variance(
                        input.history_extent,
                        input.history_variance,
                        input.history_tile_mask,
                        history_tiles,
                        input.history_view_projection,
                        pixel.hit_world,
                    );
                    let mix_variance = if history_variance[1] == 0.0 {
                        0.0
                    } else {
                        0.85
                    };
                    let history_variance_value = history_variance[0].clamp(0.0, 100.0);
                    let out_variance = mix(
                        in_variance * in_variance,
                        history_variance_value * history_variance_value,
                        mix_variance,
                    )
                    .sqrt();
                    variance[index] = quantize_f16(out_variance);
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct BilateralPixel {
    pub closure_present: bool,
    /// Post-reverse-Z depth; zero denotes the background.
    pub depth: f32,
    pub apparent_roughness: f32,
    pub world_position: [f32; 3],
    /// Closure-decoded normal transformed to world space for bilateral filtering.
    pub world_normal: [f32; 3],
    /// Interleaved-gradient-noise plus the source ray-trace RNG value.
    pub noise: [f32; 2],
}

#[derive(Clone, Copy)]
pub struct BilateralInput<'a> {
    pub extent: Extent,
    pub tile_mask: &'a [u8],
    pub pixels: &'a [BilateralPixel],
    /// Decoded temporally stabilized `UFLOAT_11_11_10` radiance and `SFLOAT_16` variance.
    pub temporal_radiance: &'a [[f32; 3]],
    pub temporal_variance: &'a [f32],
}

pub struct BilateralOutput<'a> {
    /// Decoded values quantized to `UFLOAT_11_11_10` at each source image write.
    pub radiance: &'a mut [[f32; 3]],
}

/// EEVEE filter helpers referenced by the pinned bilateral shader.
pub trait BilateralKernelOps: ColorTransfer {
    fn gaussian_factor(&self, filter_size: f32, sigma: f32) -> f32;
    fn gaussian_weight(&self, factor: f32, distance_squared: f32) -> f32;
    fn planar_weight(
        &self,
        center_normal: [f32; 3],
        center_position: [f32; 3],
        sample_position: [f32; 3],
        scene_scale: f32,
    ) -> f32;
    fn angle_weight(&self, center_normal: [f32; 3], sample_normal: [f32; 3]) -> f32;
}

/// Executes the pinned bilateral pass in tile, sample, and accumulation order.
pub fn bilateral_denoise(
    input: BilateralInput<'_>,
    output: BilateralOutput<'_>,
    ops: &impl BilateralKernelOps,
) -> Result<()> {
    let BilateralOutput { radiance } = output;
    let pixel_count = input.extent.pixel_count()?;
    let tile_extent = input.extent.tile_extent()?;
    check_len(
        "bilateral tile mask",
        input.tile_mask.len(),
        tile_extent.pixel_count()?,
    )?;
    check_len("bilateral pixels", input.pixels.len(), pixel_count)?;
    check_len(
        "bilateral radiance",
        input.temporal_radiance.len(),
        pixel_count,
    )?;
    check_len(
        "bilateral variance",
        input.temporal_variance.len(),
        pixel_count,
    )?;
    check_len("bilateral radiance output", radiance.len(), pixel_count)?;

    for tile_y in 0..tile_extent.height {
        for tile_x in 0..tile_extent.width {
            let tile_index = tile_extent.index(tile_x, tile_y).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "bilateral tile index overflow")
            })?;
            if input.tile_mask[tile_index] & 1 == 0 {
                continue;
            }
            let x_end = ((tile_x + 1) * RAYTRACE_GROUP_SIZE).min(input.extent.width);
            let y_end = ((tile_y + 1) * RAYTRACE_GROUP_SIZE).min(input.extent.height);
            for y in tile_y * RAYTRACE_GROUP_SIZE..y_end {
                for x in tile_x * RAYTRACE_GROUP_SIZE..x_end {
                    let index = input.extent.index(x, y).ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "bilateral pixel index overflow")
                    })?;
                    let center = input.pixels[index];
                    if !center.closure_present {
                        continue;
                    }
                    let variance = input.temporal_variance[index];
                    let in_radiance = input.temporal_radiance[index];
                    let is_background = center.depth == 0.0;
                    let is_smooth = center.apparent_roughness < 0.05;
                    let is_low_variance = variance < 0.1;
                    let filter_size_factor =
                        saturate(center.apparent_roughness * 8.0) * saturate(variance * 2.0);
                    let filter_size = 9.0 * filter_size_factor;
                    let sample_count = (1.0 + 14.0 * saturate(variance * 2.0)) as u32;
                    if is_smooth || is_background || is_low_variance {
                        radiance[index] = quantize_radiance(in_radiance);
                        continue;
                    }
                    let mut accumulated = ops.log_from_scene_linear(in_radiance);
                    let mut accumulated_weight = 1.0;
                    for sample in 0..sample_count {
                        let hammersley = hammersley_2d(sample, sample_count);
                        let offset_float = [
                            fract(hammersley[0] + center.noise[0]) - 0.5,
                            fract(hammersley[1] + center.noise[1]) - 0.5,
                        ];
                        let offset = [
                            (offset_float[0] * filter_size + 0.5).floor() as i32,
                            (offset_float[1] * filter_size + 0.5).floor() as i32,
                        ];
                        let sample_x = x as i32 + offset[0];
                        let sample_y = y as i32 + offset[1];
                        if sample_x < 0 || sample_y < 0 {
                            continue;
                        }
                        let (sample_x, sample_y) = (sample_x as u32, sample_y as u32);
                        let Some(sample_index) = input.extent.index(sample_x, sample_y) else {
                            continue;
                        };
                        let sample_tile = [
                            sample_x / RAYTRACE_GROUP_SIZE,
                            sample_y / RAYTRACE_GROUP_SIZE,
                        ];
                        let Some(sample_tile_index) =
                            tile_extent.index(sample_tile[0], sample_tile[1])
                        else {
                            continue;
                        };
                        if input.tile_mask[sample_tile_index] == 0 {
                            continue;
                        }
                        let sample_pixel = input.pixels[sample_index];
                        if sample_pixel.depth == 0.0
                            || is_invalid_radiance(input.temporal_radiance[sample_index])
                            || !sample_pixel.closure_present
                        {
                            continue;
                        }
                        let gaussian = ops.gaussian_factor(filter_size, 1.5);
                        let depth_weight = ops.planar_weight(
                            center.world_normal,
                            center.world_position,
                            sample_pixel.world_position,
                            10_000.0,
                        );
                        let spatial_weight = ops.gaussian_weight(
                            gaussian,
                            (offset[0] * offset[0] + offset[1] * offset[1]) as f32,
                        );
                        let normal_weight =
                            ops.angle_weight(center.world_normal, sample_pixel.world_normal);
                        let weight = depth_weight * spatial_weight * normal_weight;
                        let log_radiance =
                            ops.log_from_scene_linear(input.temporal_radiance[sample_index]);
                        for (accumulated_channel, log_channel) in
                            accumulated.iter_mut().zip(log_radiance)
                        {
                            *accumulated_channel += log_channel * weight;
                        }
                        accumulated_weight += weight;
                    }
                    for accumulated_channel in &mut accumulated {
                        *accumulated_channel *= safe_rcp(accumulated_weight);
                    }
                    radiance[index] = quantize_radiance(ops.scene_linear_from_log(accumulated));
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct LocalStatistics {
    mean: [f32; 3],
    deviation: [f32; 3],
    clamp_min: [f32; 3],
    clamp_max: [f32; 3],
}

fn local_statistics(
    extent: Extent,
    radiance: &[[f32; 3]],
    center: [u32; 2],
    center_radiance: [f32; 3],
    color: &impl ColorTransfer,
) -> LocalStatistics {
    let center_ycocg = color.ycocg_from_scene_linear(center_radiance);
    let mut mean = center_ycocg;
    let mut moment = square3(center_ycocg);
    let mut weight = 1.0;
    for x_offset in -1_i32..=1 {
        for y_offset in -1_i32..=1 {
            if x_offset == 0 && y_offset == 0 {
                continue;
            }
            let x = center[0] as i32 + x_offset;
            let y = center[1] as i32 + y_offset;
            if x < 0 || y < 0 {
                continue;
            }
            let Some(index) = extent.index(x as u32, y as u32) else {
                continue;
            };
            let sample = radiance[index];
            if is_invalid_radiance(sample) {
                continue;
            }
            let sample_ycocg = color.ycocg_from_scene_linear(sample);
            for channel in 0..3 {
                mean[channel] += sample_ycocg[channel];
                moment[channel] += sample_ycocg[channel] * sample_ycocg[channel];
            }
            weight += 1.0;
        }
    }
    let reciprocal = safe_rcp(weight);
    for channel in 0..3 {
        mean[channel] *= reciprocal;
        moment[channel] *= reciprocal;
    }
    let mut deviation = [0.0; 3];
    let mut clamp_min = [0.0; 3];
    let mut clamp_max = [0.0; 3];
    for channel in 0..3 {
        let variance = (moment[channel] - mean[channel] * mean[channel]).abs();
        deviation[channel] = 1.0e-4_f32.max(variance.sqrt());
        clamp_min[channel] = mean[channel] - deviation[channel];
        clamp_max[channel] = mean[channel] + deviation[channel];
    }
    LocalStatistics {
        mean,
        deviation,
        clamp_min,
        clamp_max,
    }
}

fn sample_history_radiance(
    extent: Extent,
    radiance: &[[f32; 3]],
    tile_mask: &[u8],
    tile_extent: Extent,
    view_projection: [[f32; 4]; 4],
    world_position: [f32; 3],
    local: LocalStatistics,
    color: &impl ColorTransfer,
) -> [f32; 4] {
    let Some(uv) = project_uv(view_projection, world_position) else {
        return [0.0; 4];
    };
    let texel_x = uv[0] * extent.width as f32 - 0.5;
    let texel_y = uv[1] * extent.height as f32 - 0.5;
    let base_x = texel_x.floor() as i32;
    let base_y = texel_y.floor() as i32;
    let weights = bilinear_weights([texel_x - base_x as f32, texel_y - base_y as f32]);
    let positions = [
        [base_x, base_y + 1],
        [base_x + 1, base_y + 1],
        [base_x + 1, base_y],
        [base_x, base_y],
    ];
    let mut history = [0.0; 4];
    for tap in 0..4 {
        let [x, y] = positions[tap];
        if x < 0 || y < 0 {
            continue;
        }
        let (x, y) = (x as u32, y as u32);
        let Some(index) = extent.index(x, y) else {
            continue;
        };
        let Some(tile_index) = tile_extent.index(x / RAYTRACE_GROUP_SIZE, y / RAYTRACE_GROUP_SIZE)
        else {
            continue;
        };
        if tile_mask[tile_index] == 0 {
            continue;
        }
        let sample = radiance[index];
        if is_invalid_radiance(sample) {
            continue;
        }
        for channel in 0..3 {
            history[channel] += sample[channel] * weights[tap];
        }
        history[3] += weights[tap];
    }
    let history_ycocg = color.ycocg_from_scene_linear([history[0], history[1], history[2]]);
    let mut distance = 0.0;
    for ((history, mean), deviation) in history_ycocg
        .into_iter()
        .zip(local.mean)
        .zip(local.deviation)
    {
        distance += (history - mean).abs() / deviation;
    }
    let weight = (-4.0 * distance).exp2();
    [
        history_ycocg[0] * weight,
        history_ycocg[1] * weight,
        history_ycocg[2] * weight,
        history[3] * weight,
    ]
}

fn sample_history_variance(
    extent: Extent,
    variance: &[f32],
    tile_mask: &[u8],
    tile_extent: Extent,
    view_projection: [[f32; 4]; 4],
    world_position: [f32; 3],
) -> [f32; 2] {
    let Some(uv) = project_uv(view_projection, world_position) else {
        return [0.0; 2];
    };
    if !(0.0..1.0).contains(&uv[0]) || !(0.0..1.0).contains(&uv[1]) {
        return [0.0; 2];
    }
    let texel_x = uv[0] * extent.width as f32 - 0.5;
    let texel_y = uv[1] * extent.height as f32 - 0.5;
    let base_x = texel_x.floor() as i32;
    let base_y = texel_y.floor() as i32;
    let weights = bilinear_weights([texel_x - base_x as f32, texel_y - base_y as f32]);
    let samples = [
        [base_x, base_y + 1],
        [base_x + 1, base_y + 1],
        [base_x + 1, base_y],
        [base_x, base_y],
    ];
    let mut value = 0.0;
    for tap in 0..4 {
        let [x, y] = samples[tap];
        if x < 0 || y < 0 {
            continue;
        }
        let x = (x as u32).min(extent.width - 1);
        let y = (y as u32).min(extent.height - 1);
        if let Some(index) = extent.index(x, y) {
            value += variance[index] * weights[tap];
        }
    }
    let history_x = (uv[0] * extent.width as f32).floor() as u32;
    let history_y = (uv[1] * extent.height as f32).floor() as u32;
    let Some(tile_index) = tile_extent.index(
        history_x / RAYTRACE_GROUP_SIZE,
        history_y / RAYTRACE_GROUP_SIZE,
    ) else {
        return [0.0; 2];
    };
    if tile_mask[tile_index] != 0 {
        [value, 1.0]
    } else {
        [0.0; 2]
    }
}

fn project_uv(matrix: [[f32; 4]; 4], position: [f32; 3]) -> Option<[f32; 2]> {
    let point = [position[0], position[1], position[2], 1.0];
    let mut clip = [0.0; 4];
    for row in 0..4 {
        for column in 0..4 {
            clip[row] += matrix[row][column] * point[column];
        }
    }
    if clip[3] == 0.0 || !clip[3].is_finite() {
        return None;
    }
    let uv = [clip[0] / clip[3] * 0.5 + 0.5, clip[1] / clip[3] * 0.5 + 0.5];
    (uv[0].is_finite() && uv[1].is_finite()).then_some(uv)
}

fn clear_spatial_neighbor_tiles(
    extent: Extent,
    tile_extent: Extent,
    tile_mask: &[u8],
    tile: [u32; 2],
    radiance: &mut [[f32; 3]],
    variance: &mut [f32],
    hit_depth: &mut [f32],
) {
    for dx in -1_i32..=1 {
        for dy in -1_i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let neighbor_x = tile[0] as i32 + dx;
            let neighbor_y = tile[1] as i32 + dy;
            if neighbor_x < 0 || neighbor_y < 0 {
                continue;
            }
            let (neighbor_x, neighbor_y) = (neighbor_x as u32, neighbor_y as u32);
            let Some(tile_index) = tile_extent.index(neighbor_x, neighbor_y) else {
                continue;
            };
            if tile_mask[tile_index] & 1 != 0 {
                continue;
            }
            for local_y in 0..RAYTRACE_GROUP_SIZE {
                for local_x in 0..RAYTRACE_GROUP_SIZE {
                    let x = neighbor_x * RAYTRACE_GROUP_SIZE + local_x;
                    let y = neighbor_y * RAYTRACE_GROUP_SIZE + local_y;
                    if let Some(index) = extent.index(x, y) {
                        write_invalid(index, radiance, variance, hit_depth);
                    }
                }
            }
        }
    }
}

fn clear_temporal_neighbor_variance(
    extent: Extent,
    tile_extent: Extent,
    tile_mask: &[u8],
    tile: [u32; 2],
    output: &mut [f32],
) {
    for dx in -1_i32..=1 {
        for dy in -1_i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let neighbor_x = tile[0] as i32 + dx;
            let neighbor_y = tile[1] as i32 + dy;
            if neighbor_x < 0 || neighbor_y < 0 {
                continue;
            }
            let (neighbor_x, neighbor_y) = (neighbor_x as u32, neighbor_y as u32);
            let Some(tile_index) = tile_extent.index(neighbor_x, neighbor_y) else {
                continue;
            };
            if tile_mask[tile_index] & 1 != 0 {
                continue;
            }
            for local_y in 0..RAYTRACE_GROUP_SIZE {
                for local_x in 0..RAYTRACE_GROUP_SIZE {
                    let x = neighbor_x * RAYTRACE_GROUP_SIZE + local_x;
                    let y = neighbor_y * RAYTRACE_GROUP_SIZE + local_y;
                    if let Some(index) = extent.index(x, y) {
                        output[index] = quantize_f16(0.0);
                    }
                }
            }
        }
    }
}

fn trace_coordinate(pixel: [u32; 2], offset: [u32; 2]) -> [i32; 2] {
    [
        (pixel[0] as i32 - offset[0] as i32).max(0),
        (pixel[1] as i32 - offset[1] as i32).max(0),
    ]
}

fn trace_index(extent: Extent, coordinate: [i32; 2]) -> Option<usize> {
    if coordinate[0] < 0 || coordinate[1] < 0 {
        return None;
    }
    extent.index(coordinate[0] as u32, coordinate[1] as u32)
}

fn bilinear_weights(coordinate: [f32; 2]) -> [f32; 4] {
    [
        (1.0 - coordinate[0]) * coordinate[1],
        coordinate[0] * coordinate[1],
        coordinate[0] * (1.0 - coordinate[1]),
        (1.0 - coordinate[0]) * (1.0 - coordinate[1]),
    ]
}

fn hammersley_2d(index: u32, count: u32) -> [f32; 2] {
    [index as f32 / count as f32, radical_inverse_vdc(index)]
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x00ff_00ff) << 8) | ((bits & 0xff00_ff00) >> 8);
    bits = ((bits & 0x0f0f_0f0f) << 4) | ((bits & 0xf0f0_f0f0) >> 4);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xcccc_cccc) >> 2);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xaaaa_aaaa) >> 1);
    bits as f32 * 2.328_306_4e-10
}

fn sample_disk(sample: [f32; 2]) -> [f32; 2] {
    let phi = (sample[1] - 0.5) * std::f32::consts::TAU;
    let cos_phi = phi.cos();
    let sign_phi = if phi == 0.0 { 0.0 } else { phi.signum() };
    let sin_phi = (1.0 - cos_phi * cos_phi).sqrt() * sign_phi;
    let radius = sample[0].sqrt();
    [radius * cos_phi, radius * sin_phi]
}

fn write_invalid(
    index: usize,
    radiance: &mut [[f32; 3]],
    variance: &mut [f32],
    hit_depth: &mut [f32],
) {
    radiance[index] = quantize_radiance(INVALID_RADIANCE);
    variance[index] = quantize_f16(0.0);
    hit_depth[index] = 0.0;
}

fn check_len(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(PotError::invalid_argument(format!(
            "{name} has {actual} elements; expected {expected}"
        )));
    }
    Ok(())
}

fn is_invalid_radiance(value: [f32; 3]) -> bool {
    value == INVALID_RADIANCE
}

fn safe_rcp(value: f32) -> f32 {
    // Match the EEVEE Metal path, where subnormal operands are flushed before reciprocal.
    if value.abs() < f32::MIN_POSITIVE {
        0.0
    } else {
        1.0 / value
    }
}

fn saturate(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn fract(value: f32) -> f32 {
    value - value.floor()
}

fn mix(first: f32, second: f32, factor: f32) -> f32 {
    first * (1.0 - factor) + second * factor
}

fn mix3(first: [f32; 3], second: [f32; 3], factor: f32) -> [f32; 3] {
    [
        mix(first[0], second[0], factor),
        mix(first[1], second[1], factor),
        mix(first[2], second[2], factor),
    ]
}

fn add4(first: &mut [f32; 4], second: [f32; 4]) {
    for channel in 0..4 {
        first[channel] += second[channel];
    }
}

fn add3(first: [f32; 3], second: [f32; 3]) -> [f32; 3] {
    [
        first[0] + second[0],
        first[1] + second[1],
        first[2] + second[2],
    ]
}

fn scale3(value: [f32; 3], factor: f32) -> [f32; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if length == 0.0 {
        [0.0; 3]
    } else {
        scale3(value, 1.0 / length)
    }
}

fn normalize2(value: [f32; 2]) -> [f32; 2] {
    let length = (value[0] * value[0] + value[1] * value[1]).sqrt();
    if length == 0.0 {
        [0.0; 2]
    } else {
        [value[0] / length, value[1] / length]
    }
}

fn square3(value: [f32; 3]) -> [f32; 3] {
    [
        value[0] * value[0],
        value[1] * value[1],
        value[2] * value[2],
    ]
}

fn quantize_radiance(color: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|channel| {
        let value = color[channel];
        if !value.is_finite() {
            return if value.is_sign_positive() {
                f32::INFINITY
            } else {
                0.0
            };
        }
        super::eevee_transmission::truncate_ufloat(f64::from(value), [6, 6, 5][channel]) as f32
    })
}

pub(super) fn store_f16_bits_rtz(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let significand = bits & 0x7f_ffff;
    if exponent == 0xff {
        sign | 0x7c00
            | if significand == 0 {
                0
            } else {
                ((significand >> 13) as u16).max(1)
            }
    } else {
        let half_exponent = exponent - 127 + 15;
        if half_exponent >= 0x1f {
            sign | 0x7bff
        } else if half_exponent > 0 {
            sign | ((half_exponent as u16) << 10) | (significand >> 13) as u16
        } else if half_exponent >= -10 {
            sign | (((significand | 0x80_0000) >> (14 - half_exponent)) as u16)
        } else {
            sign
        }
    }
}

fn quantize_f16(value: f32) -> f32 {
    exr::prelude::f16::from_bits(store_f16_bits_rtz(value)).to_f32()
}

#[cfg(test)]
mod tests {
    use super::{ColorTransfer, safe_rcp};

    #[test]
    fn safe_rcp_flushes_subnormal_inputs_like_eevee_metal() {
        assert_eq!(safe_rcp(f32::from_bits(1)), 0.0);
        assert_eq!(safe_rcp(-f32::from_bits(1)), 0.0);
        assert_eq!(safe_rcp(f32::MIN_POSITIVE), 1.0 / f32::MIN_POSITIVE);
    }

    #[test]
    fn safe_rcp_preserves_nan_reciprocal_semantics() {
        assert!(safe_rcp(f32::NAN).is_nan());
    }

    #[test]
    fn sample_disk_matches_source_cardinal_directions_and_zero_sign() {
        let samples = [
            ([0.25, 0.5], [0.5, 0.0]),
            ([0.25, 0.75], [0.0, 0.5]),
            ([0.25, 0.25], [0.0, -0.5]),
        ];

        for (sample, expected) in samples {
            let actual = super::sample_disk(sample);
            assert!(
                (actual[0] - expected[0]).abs() < 1.0e-6
                    && (actual[1] - expected[1]).abs() < 1.0e-6,
                "sample {sample:?} produced {actual:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn gpu_f16_stores_match_measured_signed_rtz_boundaries() {
        let vectors = [
            (f32::from_bits(0xbf4b_ffff), 0xba5f),
            (0.796_875, 0x3a60),
            (f32::from_bits(0x3f4b_ffff), 0x3a5f),
            (f32::from_bits(0x3f8f_ffff), 0x3c7f),
            (1.125, 0x3c80),
            (f32::from_bits(0x3380_0000), 0x0001),
            (f32::from_bits(0x337f_ffff), 0x0000),
            (f32::from_bits(0xb37f_ffff), 0x8000),
            (65520.0, 0x7bff),
            (f32::MAX, 0x7bff),
            (f32::INFINITY, 0x7c00),
            (f32::NEG_INFINITY, 0xfc00),
            (f32::from_bits(0x7f80_0001), 0x7c01),
        ];
        for (value, expected) in vectors {
            assert_eq!(
                super::store_f16_bits_rtz(value),
                expected,
                "input bits {:#010x}",
                value.to_bits()
            );
        }
    }

    #[derive(Clone, Copy)]
    struct TestColor;

    impl super::ColorTransfer for TestColor {
        fn log_from_scene_linear(&self, color: [f32; 3]) -> [f32; 3] {
            color.map(|channel| (1.0 + channel).log2())
        }

        fn scene_linear_from_log(&self, color: [f32; 3]) -> [f32; 3] {
            color.map(|channel| channel.exp2() - 1.0)
        }

        fn ycocg_from_scene_linear(&self, color: [f32; 3]) -> [f32; 3] {
            [
                color[0] + 2.0 * color[1] + color[2],
                2.0 * (color[0] - color[2]),
                2.0 * color[1] - color[0] - color[2],
            ]
        }

        fn scene_linear_from_ycocg(&self, color: [f32; 3]) -> [f32; 3] {
            [
                (color[0] + color[1] - color[2]) * 0.25,
                (color[0] + color[2]) * 0.25,
                (color[0] - color[1] - color[2]) * 0.25,
            ]
        }

        fn safe_color(&self, color: [f32; 3]) -> [f32; 3] {
            color.map(|channel| {
                if channel.is_nan() || channel <= 0.0 {
                    0.0
                } else if channel >= 1.0e20 {
                    1.0e20
                } else {
                    channel
                }
            })
        }
    }

    #[test]
    fn temporal_reprojection_keeps_current_radiance_when_history_weight_is_subnormal()
    -> super::Result<()> {
        use super::{Extent, TemporalInput, TemporalOutput, TemporalPixel, temporal_denoise};
        use super::{local_statistics, sample_history_radiance};

        let color = TestColor;
        let extent = Extent {
            width: 1,
            height: 1,
        };
        let tile_mask = [1];
        let spatial_radiance = [[0.02, 0.01, 0.02]];
        let spatial_variance = [0.028_396_606];
        let pixels = [TemporalPixel {
            surface_world: [0.0; 3],
            hit_world: [0.0; 3],
        }];
        let history_view_projection = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let local = local_statistics(
            extent,
            &spatial_radiance,
            [0, 0],
            spatial_radiance[0],
            &color,
        );
        let history_ycocg = [local.mean[0] + 36.5 * 1.0e-4, local.mean[1], local.mean[2]];
        let history_radiance = [color.scene_linear_from_ycocg(history_ycocg)];
        let history_variance = [0.0];
        let history_tile_mask = [1];
        let history_surface = sample_history_radiance(
            extent,
            &history_radiance,
            &history_tile_mask,
            extent,
            history_view_projection,
            pixels[0].surface_world,
            local,
            &color,
        );
        let history_hit = sample_history_radiance(
            extent,
            &history_radiance,
            &history_tile_mask,
            extent,
            history_view_projection,
            pixels[0].hit_world,
            local,
            &color,
        );
        let history_weight = history_surface[3] + history_hit[3];
        assert!(history_weight.is_subnormal());

        let mut output_radiance = [[0.0; 3]];
        let mut output_variance = [0.0];
        temporal_denoise(
            TemporalInput {
                extent,
                tile_mask: &tile_mask,
                spatial_radiance: &spatial_radiance,
                spatial_variance: &spatial_variance,
                pixels: &pixels,
                history_extent: extent,
                history_radiance: &history_radiance,
                history_variance: &history_variance,
                history_tile_mask: &history_tile_mask,
                history_view_projection,
            },
            TemporalOutput {
                radiance: &mut output_radiance,
                variance: &mut output_variance,
            },
            &color,
        )?;

        assert_eq!(
            output_radiance[0],
            super::quantize_radiance(spatial_radiance[0])
        );
        assert!(output_radiance[0].iter().all(|channel| channel.is_finite()));
        Ok(())
    }
}
