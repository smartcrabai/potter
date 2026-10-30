//! Per-hit and frame-resolve helpers for Blender 5.2.2 EEVEE advanced closures.
//!
//! EEVEE's opaque subsurface closure is not evaluated as a per-hit approximation: it records its
//! material/light data, then runs the 16-sample same-object screen-space Burley resolve after
//! lighting (`draw/engines/eevee/eevee_subsurface.cc` and `shaders/eevee_subsurface.bsl.hh`).
//! - `node_shader_bsdf_refraction.cc`, `gpu_shader_material_refraction.glsl`, and `eevee_closure.bsl.hh` for
//!   refraction closure setup and light-probe direction.
//! - `node_shader_bsdf_translucent.cc`, `gpu_shader_material_translucent.glsl`, and
//!   `eevee_bxdf_diffuse.bsl.hh` for the Lambert and translucent per-hit responses.
//! - `node_shader_bsdf_metallic.cc`, `gpu_shader_material_metallic.glsl`, and
//!   `eevee_bxdf_lut_lib.bsl.hh` for the conductor Fresnel inputs and metallic closure.
//! - `node_shader_holdout.cc`, `gpu_shader_material_holdout.glsl`, `eevee_surf_forward.bsl.hh`, and
//!   `eevee_forward_resolve.bsl.hh` for holdout alpha/transmittance separation.

use std::f32::consts::{PI, TAU};

use glam::DVec3;

use super::eevee_transmission::Thickness;
use crate::shader::{BsdfModel, BsdfParams, MetallicFresnel, ShaderClosure, SubsurfaceMethod};

const SSS_SAMPLE_COUNT: usize = 16;
const SSS_BURLEY_TRUNCATE_CDF: f32 = 0.996_379;
const SSS_BURLEY_TRUNCATE: f32 = 16.0;
const SSS_MIN_RADIUS: f32 = 1.0e-4;
const RADIANCE_MANTISSA_BITS: [u32; 3] = [6, 6, 5];

/// EEVEE's closure data needed by the screen-space subsurface pass.
///
/// The G-buffer carries at most three closure layers. Each layer contributes a weighted profile while
/// `direct` and `indirect` hold the closure-selection-weighted neighborhood radiance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EeveeSubsurfaceSample {
    pub(crate) closures: [Option<EeveeSubsurfaceClosure>; MAX_EEVEE_SUBSURFACE_CLOSURES],
    pub(crate) direct: [f32; 3],
    pub(crate) indirect: [f32; 3],
    pub(crate) object_index: u32,
    pub(crate) depth: f64,
    pub(crate) split_radiance: bool,
}

pub(crate) const MAX_EEVEE_SUBSURFACE_CLOSURES: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EeveeSubsurfaceClosure {
    pub(crate) color: [f32; 3],
    pub(crate) radius: [f32; 3],
}

/// One pixel's SSS closure and its front-to-back contribution weight in the Combined pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EeveeSubsurfacePixel {
    pub(crate) sample: EeveeSubsurfaceSample,
    pub(crate) weight: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum EeveeSubsurfaceProjection {
    Perspective {
        horizontal_scale: f64,
        vertical_scale: f64,
    },
    Orthographic {
        half_width: f64,
        half_height: f64,
    },
}

/// The raster camera inputs required to reproduce EEVEE's view-space disk resolve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EeveeSubsurfaceFrame {
    pub(crate) camera_position: DVec3,
    pub(crate) forward: DVec3,
    pub(crate) right: DVec3,
    pub(crate) up: DVec3,
    pub(crate) near: f64,
    pub(crate) far: f64,
    pub(crate) shift: [f64; 2],
    pub(crate) projection: EeveeSubsurfaceProjection,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// The right eye's first column for side-by-side stereo; both eyes resolve independently.
    pub(crate) eye_split: Option<u32>,
}

#[derive(Clone, Copy)]
struct ProfileSample {
    x: f32,
    y: f32,
    pdf_inverse: f32,
}

#[derive(Clone, Copy, Default)]
struct NeighborhoodSample {
    radiance: [f32; 3],
    depth: f32,
    object_index: u32,
    valid: bool,
    has_sss: bool,
}

/// EEVEE's saturated forward `g_holdout`: Holdout closure weights are summed separately from
/// physical transparency, then clamped when the forward surface writes its film targets.
pub(crate) fn eevee_holdout_weight(closures: &[ShaderClosure]) -> f64 {
    closures
        .iter()
        .filter(|closure| matches!(closure.params.model, BsdfModel::Holdout))
        .map(|closure| closure.weight)
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// The metallic node's EEVEE input Fresnel pair after its conductor conversion and [0, 1] clamp.
pub(crate) fn eevee_metallic_fresnel(params: &BsdfParams) -> Option<([f64; 3], [f64; 3])> {
    let BsdfModel::Metallic(metallic) = params.model else {
        return None;
    };
    let (f0, f82) = match metallic.fresnel {
        MetallicFresnel::F82 => (params.base_color[..3].try_into().ok()?, metallic.edge_tint),
        MetallicFresnel::PhysicalConductor => (
            conductor_fresnel(1.0, metallic.conductor_ior, metallic.extinction),
            conductor_fresnel(1.0 / 7.0, metallic.conductor_ior, metallic.extinction),
        ),
    };
    Some((
        f0.map(|value| value.clamp(0.0, 1.0)),
        f82.map(|value| value.clamp(0.0, 1.0)),
    ))
}

/// EEVEE's `brdf_f82_tint_lut`: the view-dependent metallic reflectance used by direct and probe lighting.
pub(crate) fn eevee_metallic_brdf(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
) -> Option<DVec3> {
    let (f0, f82) = eevee_metallic_fresnel(params)?;
    let n_dot_v = normal.dot(view).clamp(0.0, 1.0);
    let roughness = params.roughness.clamp(0.0, 1.0);
    let scale = metallic_brdf_channel(0, roughness, n_dot_v);
    let bias = metallic_brdf_channel(1, roughness, n_dot_v);
    let metal_bias = metallic_brdf_channel(2, roughness, n_dot_v);

    let (f5, f6) = {
        let f: f64 = 6.0 / 7.0;
        (f.powi(5), f.powi(6))
    };
    let f_schlick = DVec3::from_array(f0).lerp(DVec3::ONE, f5);
    let f82_tint = f_schlick * (7.0 / f6) * (DVec3::ONE - DVec3::from_array(f82));
    let mut reflectance = DVec3::from_array(f0) * scale + DVec3::splat(bias);
    if let BsdfModel::Metallic(metallic) = params.model
        && metallic.distribution == crate::shader::MicrofacetDistribution::MultiGgx
    {
        let ess = scale + bias;
        let ems = 1.0 - ess;
        let f_avg = DVec3::from_array(f0) + (DVec3::ONE - DVec3::from_array(f0)) / 21.0;
        reflectance /= DVec3::ONE - f_avg * ems;
    }
    Some(reflectance - f82_tint * metal_bias)
}

fn metallic_brdf_channel(channel: usize, roughness: f64, n_dot_v: f64) -> f64 {
    use super::eevee_transmission::{blend, texel_weights};

    let (x0, x1, fx) = texel_weights(roughness);
    let (y0, y1, fy) = texel_weights((1.0 - n_dot_v).sqrt());
    let texel = |x, y| super::eevee_transmission_tables::brdf(x, y, channel);
    blend(
        blend(texel(x0, y0), texel(x1, y0), fx),
        blend(texel(x0, y1), texel(x1, y1), fx),
        fy,
    )
}

/// `fresnel_conductor` from `gpu_shader_material_metallic.glsl`.
#[expect(
    clippy::manual_midpoint,
    reason = "Blender's metallic node averages the two Fresnel polarizations with this exact expression"
)]
fn conductor_fresnel(cosine: f64, ior: [f64; 3], extinction: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|channel| {
        let eta = ior[channel];
        let k = extinction[channel];
        let cos2 = cosine * cosine;
        let eta2_k2 = eta * eta + k * k;
        let two_eta_cos = 2.0 * eta * cosine;
        let tmp = eta2_k2 * cos2;
        let parallel = (tmp - two_eta_cos + 1.0) / (tmp + two_eta_cos + 1.0);
        let perpendicular = (eta2_k2 - two_eta_cos + cos2) / (eta2_k2 + two_eta_cos + cos2);
        0.5 * (parallel + perpendicular)
    })
}

/// EEVEE's Principled and standalone Subsurface radius, including its legacy `4π` random-walk factor.
#[expect(
    clippy::float_cmp,
    reason = "Blender's Principled subsurface gate compares these weights exactly against one"
)]
pub(crate) fn eevee_subsurface_radius(params: &BsdfParams) -> Option<[f64; 3]> {
    let has_subsurface = match params.model {
        BsdfModel::Subsurface => true,
        BsdfModel::Principled => {
            params.subsurface_weight != 0.0
                && !params.thin_wall
                && (params.sheen_weight != 0.0
                    || (params.metallic != 1.0 && params.transmission != 1.0))
        }
        _ => false,
    };
    if !has_subsurface {
        return None;
    }
    let scale = if params.subsurface_method == SubsurfaceMethod::RandomWalk {
        4.0 * f64::from(PI)
    } else {
        1.0
    };
    Some(
        params
            .subsurface_radius
            .map(|radius| (radius * params.subsurface_scale * scale).max(0.0)),
    )
}
/// EEVEE's SSS profile after its G-buffer `UNORM_10_10_10_2` pack/unpack round trip.
pub(crate) fn eevee_subsurface_gbuffer_profile(
    color: [f64; 3],
    radius: [f64; 3],
) -> ([f32; 3], [f32; 3]) {
    let color = closure_color_roundtrip(color.map(|value| value as f32));
    let radius = radius.map(|value| ior_pack(value as f32));
    let radius = closure_color_roundtrip(radius).map(ior_unpack);
    (color, radius)
}

/// EEVEE stores octahedral normals in an `UNORM_16_16` G-buffer layer.
pub(crate) fn eevee_gbuffer_normal(normal: DVec3) -> DVec3 {
    let normal = normal.to_array().map(|value| value as f32);
    let manhattan_length = normal.iter().map(|value| value.abs()).sum::<f32>();
    let normal = normal.map(|value| value / manhattan_length);
    let sign = |value: f32| if value < 0.0 { -1.0 } else { 1.0 };
    let mut packed = if normal[2] >= 0.0 {
        [normal[0], normal[1]]
    } else {
        [
            (1.0 - normal[1].abs()) * sign(normal[0]),
            (1.0 - normal[0].abs()) * sign(normal[1]),
        ]
    };
    packed = packed.map(|value| quantize_unorm(value * 0.5 + 0.5, 65_535));
    let x = packed[0] * 2.0 - 1.0;
    let y = packed[1] * 2.0 - 1.0;
    let z = 1.0 - x.abs() - y.abs();
    let fold = (-z).clamp(0.0, 1.0);
    let x = x + if x >= 0.0 { -fold } else { fold };
    let y = y + if y >= 0.0 { -fold } else { fold };
    let length = (x * x + y * y + z * z).sqrt();
    DVec3::new(
        f64::from(x / length),
        f64::from(y / length),
        f64::from(z / length),
    )
}

fn closure_color_roundtrip(color: [f32; 3]) -> [f32; 3] {
    let maximum = color[0].max(color[1]).max(color[2]);
    let exponent = if maximum > 4.0 {
        3
    } else if maximum > 2.0 {
        2
    } else {
        i32::from(maximum > 1.0)
    };
    let scale = 2.0_f32.powi(exponent);
    color.map(|value| quantize_unorm(value / scale, 1023) * scale)
}

#[expect(
    clippy::float_cmp,
    reason = "EEVEE's G-buffer radius codec preserves IOR 1.0 with an exact sentinel"
)]
fn ior_pack(value: f32) -> f32 {
    const SCALE: f32 = 511.0 / 1023.0;
    const ONE: f32 = 512.0 / 1023.0;
    const BIAS: f32 = 510.0 / 1023.0;
    if value == 1.0 {
        ONE
    } else if value > 1.0 {
        1.0 - BIAS / value
    } else {
        SCALE * value
    }
}

fn ior_unpack(value: f32) -> f32 {
    const SCALE: f32 = 511.0 / 1023.0;
    const ONE: f32 = 512.0 / 1023.0;
    const BIAS: f32 = 510.0 / 1023.0;
    if value == ONE {
        1.0
    } else if value > ONE {
        BIAS / (1.0 - value)
    } else {
        value / SCALE
    }
}

fn quantize_unorm(value: f32, maximum: u32) -> f32 {
    let maximum = maximum as f32;
    (value.clamp(0.0, 1.0) * maximum).round() / maximum
}

/// EEVEE's `closure_light_new` reflection ray direction for a metallic GGX closure.
pub(crate) fn eevee_metallic_probe_direction(normal: DVec3, view: DVec3, roughness: f64) -> DVec3 {
    super::eevee_transmission::reflection_direction(normal, view, roughness.clamp(0.0, 1.0))
}

/// EEVEE's `bxdf_ggx_lightprobe_transmission` direction for a standalone Refraction closure.
pub(crate) fn eevee_refraction_probe_direction(
    normal: DVec3,
    view: DVec3,
    ior: f64,
    roughness: f64,
    thickness: Thickness,
) -> DVec3 {
    super::eevee_transmission::transmission_direction(
        normal,
        view,
        ior.max(1.0e-5),
        roughness.clamp(0.0, 1.0),
        thickness,
    )
}

/// Runs EEVEE's 5.2.2 16-sample Burley screen-space pass for one TAA frame.
///
/// `sample_color` is the raw, associated Combined-frame input (before RGBA16F storage). This pass
/// adds the convolved SSS closure contribution in-place; alpha and non-SSS color are unchanged.
pub(crate) fn resolve_screen_space_subsurface(
    samples: &[Option<EeveeSubsurfacePixel>],
    sample_color: &mut [[f32; 4]],
    frame: EeveeSubsurfaceFrame,
    sample_index: u32,
    pixel_jitter: [f64; 2],
) {
    if frame.width == 0 || frame.height == 0 || samples.len() != sample_color.len() {
        return;
    }
    let sample_locations = burley_sample_locations(sample_index);
    let width = frame.width as usize;
    let height = frame.height as usize;
    let eye_split = frame.eye_split.map(|split| split as usize);
    for y in 0..height {
        for x in 0..width {
            let pixel = y * width + x;
            let Some(center) = samples[pixel] else {
                continue;
            };
            let Some((eye_start, eye_width, local_x)) = eye_for_column(x, width, eye_split) else {
                continue;
            };
            for closure in center.sample.closures.iter().flatten().copied() {
                resolve_subsurface_closure(
                    center,
                    closure,
                    samples,
                    &mut sample_color[pixel],
                    frame,
                    &sample_locations,
                    pixel_jitter,
                    eye_start,
                    eye_width,
                    height,
                    y,
                    local_x,
                );
            }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "one EEVEE SSS profile needs its screen, depth, and projection inputs"
)]
fn resolve_subsurface_closure(
    center: EeveeSubsurfacePixel,
    closure: EeveeSubsurfaceClosure,
    samples: &[Option<EeveeSubsurfacePixel>],
    sample_color: &mut [f32; 4],
    frame: EeveeSubsurfaceFrame,
    sample_locations: &[ProfileSample; SSS_SAMPLE_COUNT],
    pixel_jitter: [f64; 2],
    eye_start: usize,
    eye_width: usize,
    height: usize,
    y: usize,
    local_x: usize,
) {
    let max_radius = closure.radius.iter().copied().fold(0.0_f32, f32::max);
    let depth = center.sample.depth as f32;
    let homogeneous = match frame.projection {
        EeveeSubsurfaceProjection::Perspective { .. } => depth,
        EeveeSubsurfaceProjection::Orthographic { .. } => 1.0,
    };
    if max_radius <= 0.0 || depth <= 0.0 || homogeneous <= 0.0 {
        add_unblurred(center, closure, sample_color);
        return;
    }
    let (projection_x, projection_y) = projection_coefficients(frame.projection);
    let scale_x = projection_x * 0.5 * max_radius / homogeneous;
    let scale_y = projection_y * 0.5 * max_radius / homogeneous;
    let pixel_footprint = scale_x * eye_width as f32;
    if !pixel_footprint.is_finite() || pixel_footprint <= 1.0 {
        add_unblurred(center, closure, sample_color);
        return;
    }

    let min_radius = sample_locations
        .iter()
        .map(|sample| (sample.x * sample.x + sample.y * sample.y).sqrt())
        .fold(f32::INFINITY, f32::min)
        .max(SSS_MIN_RADIUS);
    let clamped_radius = std::array::from_fn(|channel| {
        (closure.radius[channel] / max_radius).max(min_radius) * max_radius
    });
    let max_color = closure.color.iter().copied().fold(0.0_f32, f32::max);
    let albedo = closure.color.map(|value| value / max_color.max(1.0e-6));
    let profile_d = burley_setup(clamped_radius, albedo);
    let center_uv = [
        (local_x as f32 + 0.5) / eye_width as f32,
        (height as f32 - y as f32 - 0.5) / height as f32,
    ];
    let bottom_y = height - 1 - y;
    let rotation_noise = interleaved_gradient_noise(local_x as f32, bottom_y as f32);
    let theta = rotation_noise * (PI * (3.0 - 5.0_f32.sqrt()));
    let (sin_theta, cos_theta) = theta.sin_cos();
    let mut total_direct = [0.0_f32; 3];
    let mut total_indirect = [0.0_f32; 3];
    let mut total_weight = [0.0_f32; 3];

    for profile_sample in sample_locations.iter().copied() {
        let offset_x = scale_x * (cos_theta * profile_sample.x - sin_theta * profile_sample.y);
        let offset_y = scale_y * (sin_theta * profile_sample.x + cos_theta * profile_sample.y);
        let sample_uv = [center_uv[0] + offset_x, center_uv[1] + offset_y];
        let neighborhood = sample_neighborhood(
            samples,
            frame,
            eye_start,
            eye_width,
            height,
            sample_uv,
            center.sample.split_radiance,
        );
        if !neighborhood.valid
            || !neighborhood.has_sss
            || neighborhood.object_index != center.sample.object_index
        {
            continue;
        }
        let sample_position = view_position(
            frame,
            sample_uv,
            f64::from(neighborhood.depth),
            pixel_jitter,
            eye_width,
            height,
        );
        let center_position = view_position(
            frame,
            center_uv,
            f64::from(depth),
            pixel_jitter,
            eye_width,
            height,
        );
        let distance = (sample_position - center_position).length() as f32;
        let weight =
            burley_eval(profile_d, distance).map(|value| value * profile_sample.pdf_inverse);
        for channel in 0..3 {
            total_direct[channel] += neighborhood.radiance[channel] * weight[channel];
            total_weight[channel] += weight[channel];
        }
        if center.sample.split_radiance {
            let indirect = sample_indirect_radiance(
                samples,
                frame.width as usize,
                eye_start,
                eye_width,
                height,
                sample_uv,
            );
            for channel in 0..3 {
                total_indirect[channel] += indirect[channel] * weight[channel];
            }
        }
    }
    for channel in 0..3 {
        if total_weight[channel] > 0.0 {
            total_direct[channel] /= total_weight[channel];
            total_indirect[channel] /= total_weight[channel];
        }
    }
    let radiance = if center.sample.split_radiance {
        let direct = decode_rgb9e5(encode_rgb9e5(total_direct));
        let indirect: [f32; 3] = std::array::from_fn(|channel| {
            quantize_unsigned_float(total_indirect[channel], RADIANCE_MANTISSA_BITS[channel])
        });
        std::array::from_fn(|channel| direct[channel] + indirect[channel])
    } else {
        decode_rgb9e5(encode_rgb9e5(total_direct))
    };
    add_radiance(center, closure, radiance, sample_color);
}

fn eye_for_column(x: usize, width: usize, split: Option<usize>) -> Option<(usize, usize, usize)> {
    if let Some(split) = split {
        if split == 0 || split >= width {
            return None;
        }
        if x < split {
            Some((0, split, x))
        } else {
            Some((split, width - split, x - split))
        }
    } else {
        Some((0, width, x))
    }
}

fn projection_coefficients(projection: EeveeSubsurfaceProjection) -> (f32, f32) {
    match projection {
        EeveeSubsurfaceProjection::Perspective {
            horizontal_scale,
            vertical_scale,
        } => (
            (1.0 / horizontal_scale) as f32,
            (1.0 / vertical_scale) as f32,
        ),
        EeveeSubsurfaceProjection::Orthographic {
            half_width,
            half_height,
        } => ((1.0 / half_width) as f32, (1.0 / half_height) as f32),
    }
}

fn view_position(
    frame: EeveeSubsurfaceFrame,
    uv: [f32; 2],
    depth: f64,
    pixel_jitter: [f64; 2],
    eye_width: usize,
    height: usize,
) -> DVec3 {
    let ndc_x = 2.0 * f64::from(uv[0]) - 1.0
        + 2.0 * frame.shift[0]
        + 2.0 * (pixel_jitter[0] - 0.5) / eye_width as f64;
    let ndc_y = 2.0 * f64::from(uv[1]) - 1.0
        + 2.0 * frame.shift[1]
        + 2.0 * (0.5 - pixel_jitter[1]) / height as f64;
    let (x, y) = match frame.projection {
        EeveeSubsurfaceProjection::Perspective {
            horizontal_scale,
            vertical_scale,
        } => (
            ndc_x * depth * horizontal_scale,
            ndc_y * depth * vertical_scale,
        ),
        EeveeSubsurfaceProjection::Orthographic {
            half_width,
            half_height,
        } => (ndc_x * half_width, ndc_y * half_height),
    };
    frame.camera_position + frame.right * x + frame.up * y + frame.forward * depth
}

fn sample_neighborhood(
    pixels: &[Option<EeveeSubsurfacePixel>],
    frame: EeveeSubsurfaceFrame,
    eye_start: usize,
    eye_width: usize,
    height: usize,
    uv: [f32; 2],
    split_radiance: bool,
) -> NeighborhoodSample {
    let Some(sample_pixel) =
        nearest_sample_pixel(frame.width as usize, height, eye_start, eye_width, uv)
    else {
        return NeighborhoodSample::default();
    };
    let source = pixels[sample_pixel];
    NeighborhoodSample {
        radiance: source.map_or([0.0; 3], |source| {
            source_radiance(source.sample, split_radiance)
        }),
        depth: source.map_or(0.0, |source| source.sample.depth as f32),
        object_index: source.map_or(0, |source| source.sample.object_index),
        valid: true,
        has_sss: source.is_some(),
    }
}

fn sample_indirect_radiance(
    pixels: &[Option<EeveeSubsurfacePixel>],
    width: usize,
    eye_start: usize,
    eye_width: usize,
    height: usize,
    uv: [f32; 2],
) -> [f32; 3] {
    let Some(sample_pixel) = nearest_sample_pixel(width, height, eye_start, eye_width, uv) else {
        return [0.0; 3];
    };
    pixels[sample_pixel].map_or([0.0; 3], |source| quantize_radiance(source.sample.indirect))
}

/// EEVEE's SSS textures use unfiltered sampling to keep surface IDs and radiance aligned.
fn nearest_sample_pixel(
    width: usize,
    height: usize,
    eye_start: usize,
    eye_width: usize,
    uv: [f32; 2],
) -> Option<usize> {
    let x = (uv[0] * eye_width as f32).floor() as i64;
    let y = (uv[1] * height as f32).floor() as i64;
    if x < 0 || y < 0 || x >= eye_width as i64 || y >= height as i64 {
        return None;
    }
    Some((height - 1 - y as usize) * width + eye_start + x as usize)
}
fn add_unblurred(
    center: EeveeSubsurfacePixel,
    closure: EeveeSubsurfaceClosure,
    color: &mut [f32; 4],
) {
    let direct = decode_rgb9e5(encode_rgb9e5(center.sample.direct));
    let indirect = quantize_radiance(center.sample.indirect);
    let radiance = std::array::from_fn(|channel| direct[channel] + indirect[channel]);
    add_radiance(center, closure, radiance, color);
}

fn add_radiance(
    center: EeveeSubsurfacePixel,
    closure: EeveeSubsurfaceClosure,
    radiance: [f32; 3],
    color: &mut [f32; 4],
) {
    for channel in 0..3 {
        color[channel] += (f64::from(closure.color[channel])
            * center.weight[channel]
            * f64::from(radiance[channel])) as f32;
    }
}

fn source_radiance(sample: EeveeSubsurfaceSample, split_radiance: bool) -> [f32; 3] {
    let direct = decode_rgb9e5(encode_rgb9e5(sample.direct));
    let indirect = quantize_radiance(sample.indirect);
    if split_radiance {
        std::array::from_fn(|channel| {
            quantize_unsigned_float(direct[channel], RADIANCE_MANTISSA_BITS[channel])
        })
    } else {
        std::array::from_fn(|channel| {
            quantize_unsigned_float(
                direct[channel] + indirect[channel],
                RADIANCE_MANTISSA_BITS[channel],
            )
        })
    }
}

fn quantize_radiance(value: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|channel| {
        quantize_unsigned_float(value[channel], RADIANCE_MANTISSA_BITS[channel])
    })
}

fn burley_sample_locations(sample_index: u32) -> [ProfileSample; SSS_SAMPLE_COUNT] {
    let sample_u = halton(2, u64::from(sample_index) * 5 + 1) as f32;
    let sample_v = halton(3, u64::from(sample_index) * 5 + 1) as f32;
    let d = burley_setup([1.0; 3], [1.0; 3])[0];
    let golden_angle = PI * (3.0 - 5.0_f32.sqrt());
    std::array::from_fn(|index| {
        let theta = golden_angle * index as f32 + TAU * sample_u;
        let radius = burley_sample(d, (sample_v + index as f32) / SSS_SAMPLE_COUNT as f32);
        let pdf_inverse = 1.0 / burley_pdf(d, radius);
        let (sin_theta, cos_theta) = theta.sin_cos();
        ProfileSample {
            x: cos_theta * radius,
            y: sin_theta * radius,
            pdf_inverse,
        }
    })
}

fn burley_setup(radius: [f32; 3], albedo: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|channel| {
        let a = albedo[channel];
        let s = 1.9 - a + 3.5 * (a - 0.8) * (a - 0.8);
        (0.25 / PI * radius[channel]) / s
    })
}

fn burley_sample(d: f32, x: f32) -> f32 {
    let x = x * SSS_BURLEY_TRUNCATE_CDF;
    let mut radius = if x <= 0.9 {
        (x * x * 2.4).exp() - 1.0
    } else {
        15.0
    };
    for _ in 0..10 {
        let exp_r_3 = (-radius / 3.0).exp();
        let exp_r = exp_r_3 * exp_r_3 * exp_r_3;
        let f = 1.0 - 0.25 * exp_r - 0.75 * exp_r_3 - x;
        let f_derivative = 0.25 * exp_r + 0.25 * exp_r_3;
        if f.abs() < 1.0e-6 || f_derivative == 0.0 {
            break;
        }
        radius = (radius - f / f_derivative).max(0.0);
    }
    radius * d
}

fn burley_pdf(d: f32, radius: f32) -> f32 {
    let exp_r_3_d = (-radius / (3.0 * d)).exp();
    let exp_r_d = exp_r_3_d * exp_r_3_d * exp_r_3_d;
    (exp_r_d + exp_r_3_d) / (8.0 * PI * d * SSS_BURLEY_TRUNCATE_CDF)
}

fn burley_eval(d: [f32; 3], radius: f32) -> [f32; 3] {
    std::array::from_fn(|channel| {
        if radius >= SSS_BURLEY_TRUNCATE * d[channel] {
            0.0
        } else {
            let exp_r_3_d = (-radius / (3.0 * d[channel])).exp();
            let exp_r_d = exp_r_3_d * exp_r_3_d * exp_r_3_d;
            (exp_r_d + exp_r_3_d) / (4.0 * d[channel])
        }
    })
}

#[expect(
    clippy::excessive_precision,
    reason = "this is EEVEE's f32 interleaved-gradient-noise multiplier"
)]
fn interleaved_gradient_noise(x: f32, y: f32) -> f32 {
    fract(52.982_918_9 * fract(0.067_110_56 * x + 0.005_837_15 * y))
}

fn halton(base: u64, mut index: u64) -> f64 {
    let mut result = 0.0;
    let mut factor = 1.0 / base as f64;
    while index > 0 {
        result += factor * (index % base) as f64;
        index /= base;
        factor /= base as f64;
    }
    result
}

fn fract(value: f32) -> f32 {
    value - value.floor()
}

fn encode_rgb9e5(color: [f32; 3]) -> u32 {
    let color = color.map(|value| value.clamp(0.0, f32::from_bits(0x477f_0000)));
    let max_component = color[0].max(color[1]).max(color[2]);
    if max_component == 0.0 {
        return 0;
    }
    let exponent = ((max_component.to_bits() >> 23) as i32 - 127).max(-16) + 16;
    let mut shared_exponent = exponent;
    let mut denominator = 2.0_f32.powi(shared_exponent - 24);
    if (max_component / denominator + 0.5) as u32 == 512 {
        denominator *= 2.0;
        shared_exponent += 1;
    }
    let mantissa = color.map(|value| (value / denominator + 0.5) as u32);
    (shared_exponent as u32) << 27 | mantissa[2] << 18 | mantissa[1] << 9 | mantissa[0]
}

fn decode_rgb9e5(packed: u32) -> [f32; 3] {
    let exponent = (packed >> 27) as i32;
    let factor = 2.0_f32.powi(exponent - 24);
    [0, 9, 18].map(|shift| ((packed >> shift) & 0x1ff) as f32 * factor)
}

fn quantize_unsigned_float(value: f32, mantissa_bits: u32) -> f32 {
    if !value.is_finite() {
        return if value.is_sign_positive() {
            f32::INFINITY
        } else {
            0.0
        };
    }
    let value = value.max(0.0);
    if value == 0.0 {
        return 0.0;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32;
    let unbiased = exponent - 127;
    let mut target_exponent = unbiased + 15;
    if target_exponent <= 0 {
        let scaled = value * 2.0_f32.powi(14 + mantissa_bits as i32);
        let mantissa = (scaled + 0.5) as u32;
        return if mantissa >= (1 << mantissa_bits) {
            2.0_f32.powi(-14)
        } else {
            mantissa as f32 * 2.0_f32.powi(-14 - mantissa_bits as i32)
        };
    }
    if target_exponent >= 31 {
        return f32::INFINITY;
    }
    let shift = 23 - mantissa_bits;
    let mut mantissa = ((bits & 0x7f_ffff) + (1 << (shift - 1))) >> shift;
    if mantissa >= (1 << mantissa_bits) {
        mantissa = 0;
        target_exponent += 1;
    }
    if target_exponent >= 31 {
        f32::INFINITY
    } else {
        (1.0 + mantissa as f32 / (1 << mantissa_bits) as f32) * 2.0_f32.powi(target_exponent - 15)
    }
}
#[cfg(test)]
mod gbuffer_tests {
    use super::{eevee_gbuffer_normal, eevee_subsurface_gbuffer_profile};
    use glam::DVec3;

    #[test]
    fn subsurface_profile_matches_gbuffer_unorm_roundtrip() {
        let (color, radius) =
            eevee_subsurface_gbuffer_profile([0.72, 0.31, 0.12], [0.204, 0.0744, 0.0288]);
        let expected_color = [737.0 / 1023.0, 317.0 / 1023.0, 123.0 / 1023.0];
        let expected_radius = [104.0 / 511.0, 38.0 / 511.0, 15.0 / 511.0];
        for channel in 0..3 {
            assert!((color[channel] - expected_color[channel]).abs() < 1.0e-7);
            assert!((radius[channel] - expected_radius[channel]).abs() < 1.0e-6);
        }
    }

    #[test]
    fn shading_normal_matches_octahedral_unorm16_roundtrip() {
        let normal = eevee_gbuffer_normal(DVec3::new(0.06, 0.02, 0.998));
        let expected = DVec3::new(0.059_988_83, 0.019_985_31, 0.997_999);
        assert!(normal.distance(expected) < 2.0e-6);
    }
    #[test]
    fn burley_profile_stops_at_the_blender_support_radius() {
        let weights = super::burley_eval([0.1, 0.2, 0.3], 3.2);
        assert_eq!(weights[0], 0.0);
        assert_eq!(weights[1], 0.0);
        assert!(weights[2] > 0.0);
    }
}
