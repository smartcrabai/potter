//! CPU port of Blender 5.2.2 EEVEE screen-space ray tracing.
//!
//! Depth inputs are canonical (non-reversed) normalized device depth: near is 0, far/background
//! is 1. `HiZ` levels must already contain Blender's padded, max-reduced pyramid in that convention.
//! The raster pipeline owns world-ray generation, temporal reprojection, backface policy, planar
//! probes, and light-probe fallback; this module only traces and samples supplied screen buffers.

use crate::{
    error::{ErrorCode, PotError, Result},
    model::{EeveeFastGiMethod, EeveeRayTracingMethod, RenderSettings},
};
use exr::prelude::f16;
use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use serde_json::json;
/// Fully recovered settings for an EEVEE ray-tracing pass. `None` is reserved for explicit-off or
/// native legacy scenes; Blender-imported unknown state is an error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EeveeRayTracingSettings {
    pub(crate) use_fast_gi: bool,
    pub(crate) method: EeveeRayTracingMethod,
    pub(crate) resolution_scale: u32,
    /// EEVEE's `HiZ` pad uses the half-resolution sphere-probe capture extent.
    pub(crate) sphere_probe_render_extent: u32,
    pub(crate) screen_trace_quality: f64,
    pub(crate) screen_trace_thickness: f64,
    pub(crate) trace_max_roughness: f64,
    pub(crate) use_backface_hit: bool,
    pub(crate) backface_radiance_scale: f64,
    pub(crate) use_denoise: bool,
    pub(crate) denoise_spatial: bool,
    pub(crate) denoise_temporal: bool,
    pub(crate) denoise_bilateral: bool,
    pub(crate) fast_gi_bias: f64,
    pub(crate) fast_gi_resolution: u32,
    pub(crate) fast_gi_step_count: u32,
    pub(crate) fast_gi_ray_count: u32,
    pub(crate) fast_gi_quality: f64,
    pub(crate) fast_gi_distance: f64,
    pub(crate) fast_gi_thickness_near: f64,
    pub(crate) fast_gi_method: EeveeFastGiMethod,
    pub(crate) surface_indirect_clamp: f64,
}

fn eevee_source_clamp_value(value: f64) -> f64 {
    if value > 0.0 { value } else { 1.0e20 }
}
impl EeveeRayTracingSettings {
    pub(crate) fn from_render_settings(
        render: &RenderSettings,
        blender_imported: bool,
        scene_id: &str,
    ) -> Result<Option<Self>> {
        match render.eevee_use_raytracing {
            Some(false) => return Ok(None),
            None if !blender_imported => return Ok(None),
            None => {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "Blender-imported EEVEE realtime scene has unknown ray-tracing state; reimport the source .blend",
                    json!({
                        "feature_id":"render.eevee.ray_tracing.settings",
                        "scene_id":scene_id,
                        "setting":"eevee_use_raytracing",
                    }),
                ));
            }
            Some(true) => {}
        }

        macro_rules! required {
            ($value:expr, $name:literal) => {
                $value.ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::UnsupportedFeature,
                        format!(
                            "EEVEE ray tracing requires retained setting `{}`; {}",
                            $name,
                            if blender_imported {
                                "reimport the source .blend"
                            } else {
                                "supply the complete setting with render.update"
                            }
                        ),
                        json!({
                            "feature_id":"render.eevee.ray_tracing.settings",
                            "scene_id":scene_id,
                            "setting":$name,
                        }),
                    )
                })?
            };
        }

        Ok(Some(Self {
            use_fast_gi: required!(render.eevee_use_fast_gi, "eevee_use_fast_gi"),
            method: required!(render.eevee_raytracing_method, "eevee_raytracing_method"),
            resolution_scale: required!(
                render.eevee_raytracing_resolution_scale,
                "eevee_raytracing_resolution_scale"
            ),
            sphere_probe_render_extent: (required!(
                render.eevee_gi_cubemap_resolution,
                "eevee_gi_cubemap_resolution"
            ) / 2)
                .max(1),
            screen_trace_quality: required!(
                render.eevee_raytracing_screen_trace_quality,
                "eevee_raytracing_screen_trace_quality"
            ),
            screen_trace_thickness: required!(
                render.eevee_raytracing_screen_trace_thickness,
                "eevee_raytracing_screen_trace_thickness"
            ),
            trace_max_roughness: required!(
                render.eevee_raytracing_trace_max_roughness,
                "eevee_raytracing_trace_max_roughness"
            ),
            use_backface_hit: required!(
                render.eevee_raytracing_use_backface_hit,
                "eevee_raytracing_use_backface_hit"
            ),
            backface_radiance_scale: required!(
                render.eevee_raytracing_backface_radiance_scale,
                "eevee_raytracing_backface_radiance_scale"
            ),
            use_denoise: required!(
                render.eevee_raytracing_use_denoise,
                "eevee_raytracing_use_denoise"
            ),
            denoise_spatial: required!(
                render.eevee_raytracing_denoise_spatial,
                "eevee_raytracing_denoise_spatial"
            ),
            denoise_temporal: required!(
                render.eevee_raytracing_denoise_temporal,
                "eevee_raytracing_denoise_temporal"
            ),
            denoise_bilateral: required!(
                render.eevee_raytracing_denoise_bilateral,
                "eevee_raytracing_denoise_bilateral"
            ),
            fast_gi_bias: required!(render.eevee_fast_gi_bias, "eevee_fast_gi_bias"),
            fast_gi_resolution: required!(
                render.eevee_fast_gi_resolution,
                "eevee_fast_gi_resolution"
            ),
            fast_gi_step_count: required!(
                render.eevee_fast_gi_step_count,
                "eevee_fast_gi_step_count"
            ),
            fast_gi_ray_count: required!(render.eevee_fast_gi_ray_count, "eevee_fast_gi_ray_count"),
            fast_gi_quality: required!(render.eevee_fast_gi_quality, "eevee_fast_gi_quality"),
            fast_gi_distance: required!(render.eevee_fast_gi_distance, "eevee_fast_gi_distance"),
            fast_gi_thickness_near: required!(
                render.eevee_fast_gi_thickness_near,
                "eevee_fast_gi_thickness_near"
            ),
            fast_gi_method: required!(render.eevee_fast_gi_method, "eevee_fast_gi_method"),
            surface_indirect_clamp: eevee_source_clamp_value(required!(
                render.eevee_clamp_surface_indirect,
                "eevee_clamp_surface_indirect"
            )),
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScreenTraceError {
    InvalidCamera,
    InvalidResolution,
    InvalidRay,
    InvalidTraceParameters,
    InvalidHiZ,
    InvalidRadiance,
}

/// View and projection matrices for the eye that produced the screen buffers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScreenTraceCamera {
    view_to_clip: Mat4,
    clip_to_view: Mat4,
}

impl ScreenTraceCamera {
    pub(crate) fn new(view_to_clip: Mat4, clip_to_view: Mat4) -> Result<Self, ScreenTraceError> {
        if !matrix_is_finite(view_to_clip) || !matrix_is_finite(clip_to_view) {
            return Err(ScreenTraceError::InvalidCamera);
        }
        let camera = Self {
            view_to_clip,
            clip_to_view,
        };
        camera.near_view_z()?;
        Ok(camera)
    }

    fn is_perspective(self) -> bool {
        self.view_to_clip.w_axis.w == 0.0
    }

    fn near_view_z(self) -> Result<f32, ScreenTraceError> {
        let z = self.view_to_clip.z_axis.z;
        let w = self.view_to_clip.w_axis.z;
        let near = if self.is_perspective() {
            let denominator = z - 1.0;
            if denominator == 0.0 {
                return Err(ScreenTraceError::InvalidCamera);
            }
            -w / denominator
        } else {
            if z == 0.0 {
                return Err(ScreenTraceError::InvalidCamera);
            }
            -(w + 1.0) / z
        };
        near.is_finite()
            .then_some(near)
            .ok_or(ScreenTraceError::InvalidCamera)
    }

    fn point_screen_to_view(self, screen_position: Vec3) -> Vec3 {
        project_point(self.clip_to_view, screen_to_ndc(screen_position))
    }

    fn depth_view_to_screen(self, view_z: f32) -> f32 {
        let clip = self.view_to_clip * Vec4::new(0.0, 0.0, view_z, 1.0);
        clip.z / clip.w * 0.5 + 0.5
    }

    fn depth_screen_to_view(self, screen_depth: f32) -> f32 {
        self.point_screen_to_view(Vec3::new(0.0, 0.0, screen_depth))
            .z
    }
}

/// View-space ray in EEVEE's direction-plus-max-time form: the endpoint is `origin + direction * max_time`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScreenTraceRay {
    pub(crate) origin: Vec3,
    pub(crate) direction: Vec3,
    pub(crate) max_time: f32,
}

/// EEVEE's per-pixel minimum screen-depth thickness parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScreenThicknessParameters {
    pub(crate) ndc_scale: f32,
    pub(crate) ndc_bias: f32,
    pub(crate) vs_scale: f32,
    pub(crate) vs_bias: f32,
}

impl ScreenThicknessParameters {
    fn pixel_depth_thickness_at(self, view_z: f32) -> f32 {
        let view_thickness = view_z * self.vs_scale + self.vs_bias;
        let ndc_thickness = view_thickness / (view_z * (view_z + view_thickness));
        ndc_thickness * self.ndc_scale + self.ndc_bias
    }

    fn is_finite(self) -> bool {
        self.ndc_scale.is_finite()
            && self.ndc_bias.is_finite()
            && self.vs_scale.is_finite()
            && self.vs_bias.is_finite()
    }
}

/// Parameters corresponding to `RayTraceData` fields consumed by the screen tracer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScreenTraceParameters {
    pub(crate) full_resolution: UVec2,
    pub(crate) quality: f32,
    pub(crate) ray_thickness: ScreenThicknessParameters,
}

impl ScreenTraceParameters {
    fn inverse_resolution(self) -> Vec2 {
        Vec2::new(
            1.0 / self.full_resolution.x as f32,
            1.0 / self.full_resolution.y as f32,
        )
    }

    fn validate(self) -> Result<(), ScreenTraceError> {
        if self.full_resolution.x == 0 || self.full_resolution.y == 0 {
            return Err(ScreenTraceError::InvalidResolution);
        }
        if !self.quality.is_finite() || self.quality < 0.0 || !self.ray_thickness.is_finite() {
            return Err(ScreenTraceError::InvalidTraceParameters);
        }
        Ok(())
    }
}

/// A single `HiZ` mip, row-major from the lower-left texture origin.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HiZLevel<'a> {
    width: u32,
    height: u32,
    depth: &'a [f32],
}

impl<'a> HiZLevel<'a> {
    pub(crate) fn new(width: u32, height: u32, depth: &'a [f32]) -> Result<Self, ScreenTraceError> {
        let count = width
            .checked_mul(height)
            .and_then(|count| usize::try_from(count).ok());
        if width == 0 || height == 0 || count != Some(depth.len()) {
            return Err(ScreenTraceError::InvalidHiZ);
        }
        Ok(Self {
            width,
            height,
            depth,
        })
    }
}

/// Blender `HiZ` data: source-resolution UV scale plus levels of its already-built pyramid.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HiZBuffer<'a> {
    uv_scale: Vec2,
    levels: &'a [HiZLevel<'a>],
}

impl<'a> HiZBuffer<'a> {
    pub(crate) fn new(
        uv_scale: Vec2,
        levels: &'a [HiZLevel<'a>],
    ) -> Result<Self, ScreenTraceError> {
        if !uv_scale.is_finite()
            || uv_scale.min_element() <= 0.0
            || levels.is_empty()
            || levels.iter().any(|level| {
                level.width == 0
                    || level.height == 0
                    || level
                        .width
                        .checked_mul(level.height)
                        .and_then(|count| usize::try_from(count).ok())
                        != Some(level.depth.len())
            })
        {
            return Err(ScreenTraceError::InvalidHiZ);
        }
        Ok(Self { uv_scale, levels })
    }
}

/// LOD-0 linear-radiance texture used by `raytrace_sample_screen`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScreenRadianceBuffer<'a> {
    width: u32,
    height: u32,
    /// Blender's `SFLOAT_16_16_16_16` feedback texture after its image store.
    pixels: &'a [[f16; 4]],
}

impl<'a> ScreenRadianceBuffer<'a> {
    pub(crate) fn new(
        width: u32,
        height: u32,
        pixels: &'a [[f16; 4]],
    ) -> Result<Self, ScreenTraceError> {
        let count = width
            .checked_mul(height)
            .and_then(|count| usize::try_from(count).ok());
        if width == 0 || height == 0 || count != Some(pixels.len()) {
            return Err(ScreenTraceError::InvalidRadiance);
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }
}

/// Result of `raytrace_screen`; `hit_uv.z` is the ray's normalized screen depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScreenTraceHit {
    pub(crate) valid: bool,
    pub(crate) time: f32,
    pub(crate) hit_backface: bool,
    pub(crate) hit_uv: Vec2,
    pub(crate) hit_screen_position: Vec3,
    pub(crate) hit_view_position: Vec3,
}

/// Trace `ray` through Blender's max-depth `HiZ` pyramid.
///
/// This is `raytrace_screen` in EEVEE 5.2.2. `stride_rand` is the shader's `[0, 1]` jitter;
/// `HiZ` values must be canonical depth (not the hardware reverse-Z encoding).
pub(crate) fn raytrace_screen(
    camera: ScreenTraceCamera,
    parameters: ScreenTraceParameters,
    hiz: HiZBuffer<'_>,
    ray: ScreenTraceRay,
    stride_rand: f32,
    roughness: f32,
    allow_self_intersection: bool,
) -> Result<ScreenTraceHit, ScreenTraceError> {
    parameters.validate()?;
    if !ray.origin.is_finite()
        || !ray.direction.is_finite()
        || !ray.max_time.is_finite()
        || ray.max_time < 0.0
        || ray.direction.length_squared() == 0.0
    {
        return Err(ScreenTraceError::InvalidRay);
    }
    if !stride_rand.is_finite()
        || !(0.0..=1.0).contains(&stride_rand)
        || !roughness.is_finite()
        || !(0.0..=1.0).contains(&roughness)
    {
        return Err(ScreenTraceError::InvalidTraceParameters);
    }

    let mut ray = ray;
    if camera.is_perspective() {
        let near = camera.near_view_z()?;
        if ray.origin.z + ray.direction.z * ray.max_time > near {
            ray.max_time = ((near - ray.origin.z) / ray.direction.z).abs();
        }
    }

    let screen_ray = ScreenSpaceRay::create(camera, parameters, ray)?;
    if !allow_self_intersection && screen_ray.max_time < 1.1 {
        let screen_position = screen_ray.position_at(screen_ray.max_time);
        let view_position = camera.point_screen_to_view(Vec3::new(
            screen_position.x,
            screen_position.y,
            screen_position.z.clamp(0.0, 1.0),
        ));
        return Ok(ScreenTraceHit {
            valid: false,
            time: 0.0,
            hit_backface: false,
            hit_uv: screen_position.truncate(),
            hit_screen_position: screen_position,
            hit_view_position: view_position,
        });
    }

    let max_time = screen_ray.max_time.max(1.1);
    let mut previous_delta = 0.0;
    let mut previous_time = 0.0;
    let mut depth_sample = camera.depth_view_to_screen(ray.origin.z);
    let mut delta = depth_sample - screen_ray.origin.z;
    let lod_factor = (roughness.sqrt() * 2.0 - 0.4).clamp(0.0, 1.0);
    let mut thickness_estimator = ScreenThicknessEstimator::new(depth_sample);
    let mut previous_ray_z = screen_ray.origin.z;

    let mut t = 1.001;
    let mut time = 1.001;
    let mut hit = false;
    for iteration in 1..255 {
        if hit || time >= max_time {
            break;
        }
        let stride = 1.0 + iteration as f32 * parameters.quality;
        let lod = (stride.log2() * lod_factor).floor();
        previous_time = time;
        previous_delta = delta;
        time = (t + stride * stride_rand).min(max_time);
        t += stride;

        let screen_position = screen_ray.position_at(time);
        depth_sample = sample_hiz(hiz, screen_position.truncate(), lod);
        let sample_view_z = camera.depth_screen_to_view(depth_sample);
        let sample_min_thickness = parameters
            .ray_thickness
            .pixel_depth_thickness_at(sample_view_z);
        delta = depth_sample - screen_position.z;
        hit = thickness_estimator.intersect(
            depth_sample,
            time,
            sample_min_thickness,
            screen_position.z,
            previous_ray_z,
        );
        previous_ray_z = screen_position.z;
    }

    hit = hit && depth_sample.partial_cmp(&1.0) != Some(std::cmp::Ordering::Equal);
    time = mix(
        previous_time,
        time,
        (previous_delta / (previous_delta - delta)).clamp(0.0, 1.0),
    );
    let hit_backface = previous_delta < 0.0;
    let hit_screen_position = screen_ray.position_at(time);
    let hit_view_position = camera.point_screen_to_view(hit_screen_position);
    let trace_time = (hit_view_position - ray.origin).length() / ray.direction.length();
    let valid = hit
        && sample_hiz(hiz, hit_screen_position.truncate(), 0.0).partial_cmp(&1.0)
            != Some(std::cmp::Ordering::Equal);

    Ok(ScreenTraceHit {
        valid,
        time: trace_time,
        hit_backface,
        hit_uv: hit_screen_position.truncate(),
        hit_screen_position,
        hit_view_position,
    })
}

/// Source-faithful screen radiance sample, including EEVEE's contact-aware 4-point log blur.
/// `sample_uv` may be the trace UV or the caller's history-reprojected UV.
pub(crate) fn raytrace_sample_screen(
    camera: ScreenTraceCamera,
    parameters: ScreenTraceParameters,
    radiance: ScreenRadianceBuffer<'_>,
    hit: ScreenTraceHit,
    roughness: f32,
    sample_uv: Vec2,
) -> Result<Vec3, ScreenTraceError> {
    parameters.validate()?;
    if !roughness.is_finite() || !(0.0..=1.0).contains(&roughness) || !sample_uv.is_finite() {
        return Err(ScreenTraceError::InvalidTraceParameters);
    }

    let shading_cone_tangent = (roughness * 0.05).clamp(0.0, 1.0);
    let view_footprint = shading_cone_tangent * hit.time;
    let clip_hit = camera.view_to_clip * hit.hit_view_position.extend(1.0);
    let ndc_footprint = (view_footprint * camera.view_to_clip.x_axis.x) / clip_hit.w;
    let pixel_footprint = ndc_footprint * parameters.full_resolution.x as f32;

    if pixel_footprint < 1.0 {
        return Ok(sample_radiance(radiance, sample_uv));
    }

    let kernel_radius = (pixel_footprint - 1.0).clamp(0.0, 1.0);
    let inverse_resolution = parameters.inverse_resolution();
    let offset_x = kernel_radius * inverse_resolution.x;
    let offset_y = kernel_radius * inverse_resolution.y;
    let mut blurred = log_radiance(sample_radiance(
        radiance,
        sample_uv + Vec2::new(offset_x, offset_y),
    ));
    blurred += log_radiance(sample_radiance(
        radiance,
        sample_uv + Vec2::new(offset_x, -offset_y),
    ));
    blurred += log_radiance(sample_radiance(
        radiance,
        sample_uv + Vec2::new(-offset_x, offset_y),
    ));
    blurred += log_radiance(sample_radiance(
        radiance,
        sample_uv + Vec2::new(-offset_x, -offset_y),
    ));
    Ok(exp_radiance(blurred * 0.25) - Vec3::ONE)
}

#[derive(Clone, Copy)]
struct ScreenSpaceRay {
    origin: Vec3,
    direction: Vec3,
    max_time: f32,
}

impl ScreenSpaceRay {
    fn create(
        camera: ScreenTraceCamera,
        parameters: ScreenTraceParameters,
        ray: ScreenTraceRay,
    ) -> Result<Self, ScreenTraceError> {
        let start = camera.view_to_clip * ray.origin.extend(1.0);
        let end = camera.view_to_clip * (ray.origin + ray.direction * ray.max_time).extend(1.0);
        Self::from_start_end(start, end, parameters.inverse_resolution() * 2.0)
    }

    fn from_start_end(
        mut start: Vec4,
        mut end: Vec4,
        pixel_size: Vec2,
    ) -> Result<Self, ScreenTraceError> {
        const BIAS: f32 = -2.4e-7 * 7.0;
        start.z += BIAS;
        end.z += BIAS;
        if start.w == 0.0 || end.w == 0.0 {
            return Err(ScreenTraceError::InvalidRay);
        }
        let mut origin = start.truncate() / start.w;
        let end_position = end.truncate() / end.w;
        let mut direction = end_position - origin;
        if direction.truncate().length_squared() < 0.00001 {
            direction.x = 0.0;
            direction.y = 0.00001;
        }
        let ray_length_squared = direction.length_squared();
        let more_vertical = (direction.x / pixel_size.x).abs() < (direction.y / pixel_size.y).abs();
        let pixel_extent = if more_vertical {
            direction.y.abs()
        } else {
            direction.x.abs()
        };
        if pixel_extent == 0.0 {
            return Err(ScreenTraceError::InvalidRay);
        }
        direction /= pixel_extent;
        direction *= if more_vertical {
            pixel_size.y
        } else {
            pixel_size.x
        };
        let direction_length_squared = direction.length_squared();
        let max_time = (ray_length_squared * safe_reciprocal(direction_length_squared)).sqrt();
        let clip_distance = line_unit_box_intersection_distance_safe(origin, direction);
        let max_time = max_time.min(clip_distance);
        origin = origin * 0.5 + Vec3::splat(0.5);
        direction *= 0.5;
        if !max_time.is_finite() || !origin.is_finite() || !direction.is_finite() {
            return Err(ScreenTraceError::InvalidRay);
        }
        Ok(Self {
            origin,
            direction,
            max_time,
        })
    }

    fn position_at(self, time: f32) -> Vec3 {
        self.origin + self.direction * time
    }
}

#[derive(Clone, Copy)]
struct ScreenThicknessEstimator {
    z_slope: Vec2,
    screen_z: f32,
    time: f32,
}

impl ScreenThicknessEstimator {
    fn new(start_z: f32) -> Self {
        Self {
            z_slope: Vec2::ZERO,
            screen_z: start_z,
            time: 0.0,
        }
    }

    fn thickness(&mut self, sample_z: f32, sample_time: f32, min_thickness: f32) -> f32 {
        let delta_time = sample_time - self.time;
        let delta_z = sample_z - self.screen_z;
        let slope = delta_z / delta_time;
        let mut thickness = delta_z.abs();
        let slope_change = (self.z_slope - Vec2::splat(slope)).abs();
        if slope_change.x > self.z_slope.x.abs() || slope_change.y > self.z_slope.y.abs() {
            thickness = 0.0;
        }
        thickness = thickness.max(min_thickness);
        self.z_slope.y = self.z_slope.x;
        self.z_slope.x = slope;
        self.time = sample_time;
        self.screen_z = sample_z;
        thickness
    }

    fn intersect(
        &mut self,
        sample_z: f32,
        sample_time: f32,
        min_thickness: f32,
        ray_z: f32,
        previous_ray_z: f32,
    ) -> bool {
        let sample_thickness =
            self.thickness(sample_z, sample_time, min_thickness) + (ray_z - previous_ray_z).abs();
        ray_z >= sample_z && ray_z <= sample_z + sample_thickness
    }
}

fn sample_hiz(hiz: HiZBuffer<'_>, uv: Vec2, lod: f32) -> f32 {
    let level_index = (lod as usize).min(hiz.levels.len() - 1);
    let level = hiz.levels[level_index];
    let uv = uv * hiz.uv_scale;
    let x = (uv.x * level.width as f32)
        .floor()
        .clamp(0.0, (level.width - 1) as f32) as usize;
    let y = (uv.y * level.height as f32)
        .floor()
        .clamp(0.0, (level.height - 1) as f32) as usize;
    level.depth[y * level.width as usize + x]
}

fn sample_radiance(buffer: ScreenRadianceBuffer<'_>, uv: Vec2) -> Vec3 {
    let (x0, x1, tx) = bilinear_axis(uv.x, buffer.width);
    let (y0, y1, ty) = bilinear_axis(uv.y, buffer.height);
    let row0 = y0 as usize * buffer.width as usize;
    let row1 = y1 as usize * buffer.width as usize;
    let top = mix_vec3(
        read_radiance(buffer.pixels[row0 + x0 as usize]),
        read_radiance(buffer.pixels[row0 + x1 as usize]),
        tx,
    );
    let bottom = mix_vec3(
        read_radiance(buffer.pixels[row1 + x0 as usize]),
        read_radiance(buffer.pixels[row1 + x1 as usize]),
        tx,
    );
    mix_vec3(top, bottom, ty)
}

fn read_radiance(pixel: [f16; 4]) -> Vec3 {
    Vec3::new(pixel[0].to_f32(), pixel[1].to_f32(), pixel[2].to_f32())
}

fn bilinear_axis(uv: f32, size: u32) -> (u32, u32, f32) {
    let coordinate = uv.clamp(0.0, 1.0) * size as f32 - 0.5;
    let lower = coordinate.floor() as i32;
    let fraction = coordinate - lower as f32;
    let last = size as i32 - 1;
    (
        lower.clamp(0, last) as u32,
        (lower + 1).clamp(0, last) as u32,
        fraction,
    )
}

fn log_radiance(value: Vec3) -> Vec3 {
    Vec3::new(
        (1.0 + value.x).log2(),
        (1.0 + value.y).log2(),
        (1.0 + value.z).log2(),
    )
}

fn exp_radiance(value: Vec3) -> Vec3 {
    Vec3::new(value.x.exp2(), value.y.exp2(), value.z.exp2())
}

fn line_unit_box_intersection_distance_safe(origin: Vec3, direction: Vec3) -> f32 {
    let safe_direction = Vec3::new(
        direction.x.abs().max(1.0e-8) * if direction.x < 0.0 { -1.0 } else { 1.0 },
        direction.y.abs().max(1.0e-8) * if direction.y < 0.0 { -1.0 } else { 1.0 },
        direction.z.abs().max(1.0e-8) * if direction.z < 0.0 { -1.0 } else { 1.0 },
    );
    let first_plane = (Vec3::ONE - origin) / safe_direction;
    let second_plane = (Vec3::splat(-1.0) - origin) / safe_direction;
    first_plane.max(second_plane).min_element()
}

fn screen_to_ndc(position: Vec3) -> Vec3 {
    position * 2.0 - Vec3::ONE
}

fn project_point(matrix: Mat4, point: Vec3) -> Vec3 {
    let homogeneous = matrix * point.extend(1.0);
    homogeneous.truncate() / homogeneous.w
}

fn safe_reciprocal(value: f32) -> f32 {
    if value == 0.0 { 0.0 } else { 1.0 / value }
}

fn mix(a: f32, b: f32, factor: f32) -> f32 {
    a * (1.0 - factor) + b * factor
}

fn mix_vec3(a: Vec3, b: Vec3, factor: f32) -> Vec3 {
    a * (1.0 - factor) + b * factor
}

fn matrix_is_finite(matrix: Mat4) -> bool {
    matrix.to_cols_array().iter().all(|value| value.is_finite())
}
#[cfg(test)]
mod tests {
    use exr::prelude::f16;
    use glam::{Vec2, Vec3};

    use super::{HiZBuffer, HiZLevel, ScreenRadianceBuffer, sample_hiz, sample_radiance};

    #[test]
    fn hiz_fetch_uses_nearest_mip_texels_and_clamp_edge() -> Result<(), String> {
        let depth = [0.25, 0.75];
        let level = HiZLevel::new(2, 1, &depth).map_err(|error| format!("{error:?}"))?;
        let levels = [level];
        let hiz = HiZBuffer::new(Vec2::ONE, &levels).map_err(|error| format!("{error:?}"))?;

        assert_eq!(sample_hiz(hiz, Vec2::new(0.49, 0.5), 0.0), 0.25);
        assert_eq!(sample_hiz(hiz, Vec2::new(0.51, 0.5), 0.0), 0.75);
        assert_eq!(sample_hiz(hiz, Vec2::new(1.25, 0.5), 0.0), 0.75);
        Ok(())
    }

    #[test]
    fn radiance_fetch_bilinearly_samples_half_float_source_texels() -> Result<(), String> {
        let pixels = [
            [
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(0.0),
            ],
            [
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(0.0),
            ],
        ];
        let buffer =
            ScreenRadianceBuffer::new(2, 1, &pixels).map_err(|error| format!("{error:?}"))?;
        assert_eq!(
            sample_radiance(buffer, Vec2::new(0.5, 0.5)),
            Vec3::new(2.0, 3.0, 4.0)
        );
        Ok(())
    }
}
