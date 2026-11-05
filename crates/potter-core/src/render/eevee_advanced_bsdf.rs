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

use std::f32::consts::PI;

use glam::DVec3;

use super::eevee_transmission::Thickness;
use crate::shader::{BsdfModel, BsdfParams, MetallicFresnel, ShaderClosure, SubsurfaceMethod};

const SSS_SAMPLE_COUNT: usize = 16;
const SSS_BURLEY_TRUNCATE_CDF: f64 = 0.996_379_009_370_832_8;
const SSS_BURLEY_TRUNCATE: f32 = 16.0;
const SSS_MIN_RADIUS: f32 = 1.0e-4;
const SSS_TRANSMIT_LUT_RADIUS: f32 = 2.0;
const SSS_TRANSMIT_LUT_SIZE: f32 = 64.0;
const SSS_TRANSMIT_LUT_SCALE: f32 = 63.0 / SSS_TRANSMIT_LUT_SIZE;
const SSS_TRANSMIT_LUT_BIAS: f32 = 0.5 / SSS_TRANSMIT_LUT_SIZE;
const EEVEE_BURLEY_SSS_TRANSMITTANCE_PROFILE: [f32; 64] = [
    1.000_000, 0.957_576, 0.890_814, 0.813_161, 0.732_056, 0.652_729, 0.578_802, 0.512_565,
    0.455_203, 0.407_017, 0.365_151, 0.326_306, 0.291_256, 0.259_690, 0.231_309, 0.205_831,
    0.182_989, 0.162_537, 0.144_246, 0.127_905, 0.113_320, 0.100_316, 0.088_730, 0.078_418,
    0.069_247, 0.061_096, 0.053_858, 0.047_435, 0.041_740, 0.036_693, 0.032_225, 0.028_272,
    0.024_776, 0.021_688, 0.018_961, 0.016_556, 0.014_436, 0.012_569, 0.010_926, 0.009_482,
    0.008_213, 0.007_100, 0.006_124, 0.005_270, 0.004_523, 0.003_871, 0.003_302, 0.002_806,
    0.002_375, 0.002_002, 0.001_678, 0.001_398, 0.001_156, 0.000_948, 0.000_770, 0.000_617,
    0.000_487, 0.000_376, 0.000_283, 0.000_204, 0.000_138, 0.000_083, 0.000_037, 0.000_000,
];

/// Blender's utility-texture SSS transmittance for a stored G-buffer radius and thickness.
pub(crate) fn eevee_subsurface_transmission(radii: [f32; 3], thickness: f32) -> [f32; 3] {
    std::array::from_fn(|channel| {
        let radius = radii[channel] * SSS_TRANSMIT_LUT_RADIUS;
        if !matches!(radius.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return 0.0;
        }
        let coordinate =
            (thickness / radius).clamp(0.0, 1.0) * SSS_TRANSMIT_LUT_SCALE + SSS_TRANSMIT_LUT_BIAS;
        let position = coordinate * SSS_TRANSMIT_LUT_SIZE - 0.5;
        let low = position.floor() as usize;
        let high = (low + 1).min(EEVEE_BURLEY_SSS_TRANSMITTANCE_PROFILE.len() - 1);
        let blend = position - low as f32;
        let low = super::eevee_transmission::utility_texel(f64::from(
            EEVEE_BURLEY_SSS_TRANSMITTANCE_PROFILE[low],
        )) as f32;
        let high = super::eevee_transmission::utility_texel(f64::from(
            EEVEE_BURLEY_SSS_TRANSMITTANCE_PROFILE[high],
        )) as f32;
        low + (high - low) * blend
    })
}
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
    /// Apply G-buffer color association after the screen-space convolution.
    pub(crate) association_scale: f64,
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

#[derive(Clone, Copy)]
struct SubsurfaceResolveStack<'a> {
    samples: &'a [Option<EeveeSubsurfacePixel>],
    frame: EeveeSubsurfaceFrame,
    sample_locations: &'a [ProfileSample; SSS_SAMPLE_COUNT],
    pixel_jitter: [f64; 2],
    eye_start: usize,
    eye_width: usize,
    height: usize,
    y: usize,
    local_x: usize,
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

/// EEVEE's `brdf_f82_tint_lut`, with the explicit Fresnel inputs shared by Metallic and Principled.
///
/// `f0` and `f82` are already clamped as their node does before calling the LUT. The result is the node-stage closure
/// color before G-buffer packing; callers apply the G-buffer color roundtrip separately.
pub(crate) fn eevee_brdf_f82_tint_lut(
    f0: DVec3,
    f82: DVec3,
    cos_theta: f64,
    roughness: f64,
    do_multiscatter: bool,
) -> DVec3 {
    let n_dot_v = cos_theta.clamp(0.0, 1.0);
    let roughness = roughness.clamp(0.0, 1.0);
    let scale = metallic_brdf_channel(0, roughness, n_dot_v);
    let bias = metallic_brdf_channel(1, roughness, n_dot_v);
    let metal_bias = metallic_brdf_channel(2, roughness, n_dot_v);

    let (f5, f6) = {
        let f: f64 = 6.0 / 7.0;
        (f.powi(5), f.powi(6))
    };
    let f_schlick = f0.lerp(DVec3::ONE, f5);
    let f82_tint = f_schlick * (7.0 / f6) * (DVec3::ONE - f82);
    let mut reflectance = f0 * scale + DVec3::splat(bias);
    if do_multiscatter {
        let ess = scale + bias;
        let ems = 1.0 - ess;
        let f_avg = f0 + (DVec3::ONE - f0) / 21.0;
        reflectance /= DVec3::ONE - f_avg * ems;
    }
    reflectance - f82_tint * metal_bias
}

/// The view-dependent metallic reflectance used by direct and probe lighting.
pub(crate) fn eevee_metallic_brdf(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
) -> Option<DVec3> {
    let (f0, f82) = eevee_metallic_fresnel(params)?;
    let do_multiscatter = matches!(
        params.model,
        BsdfModel::Metallic(metallic)
            if metallic.distribution == crate::shader::MicrofacetDistribution::MultiGgx
    );
    Some(eevee_brdf_f82_tint_lut(
        DVec3::from_array(f0),
        DVec3::from_array(f82),
        normal.dot(view),
        params.roughness,
        do_multiscatter,
    ))
}

fn metallic_brdf_channel(channel: usize, roughness: f64, n_dot_v: f64) -> f64 {
    super::eevee_transmission::utility_sample_2d(roughness, (1.0 - n_dot_v).sqrt(), |x, y| {
        super::eevee_transmission_tables::brdf(x, y, channel)
    })
}

/// `fresnel_conductor` from `gpu_shader_material_metallic.glsl`.
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
        parallel.midpoint(perpendicular)
    })
}

fn f64_is_zero(value: f64) -> bool {
    value == 0.0
}

fn f64_is_one(value: f64) -> bool {
    value.to_bits() == 1.0_f64.to_bits()
}

/// EEVEE's Principled and standalone Subsurface radius, including its legacy `4*pi` random-walk factor.
pub(crate) fn eevee_subsurface_radius(params: &BsdfParams) -> Option<[f64; 3]> {
    let has_subsurface = match params.model {
        BsdfModel::Subsurface => true,
        BsdfModel::Principled => {
            !f64_is_zero(params.subsurface_weight)
                && !params.thin_wall
                && (!f64_is_zero(params.sheen_weight)
                    || (!f64_is_one(params.metallic) && !f64_is_one(params.transmission)))
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
/// The source inputs to `eevee::gbuffer::closure_data_layer_dither_round_to_nearest`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EeveeGBufferContext {
    /// `frag_co.xy`: pixel coordinates including the half-pixel center.
    pub(crate) frag_coord: [f32; 2],
    /// `sampling.rng_3D_get(SAMPLING_GBUFFER_U)`: Halton bases 11, 13 and 17 at sample + 1.
    pub(crate) sample_offset: [f32; 3],
    /// The closure-data storage layer, not the closure bin/layer index.
    pub(crate) data_layer: u32,
}

/// EEVEE's SSS profile after each of its G-buffer `UNORM_10_10_10_2` store/resolve stages.
pub(crate) fn eevee_subsurface_gbuffer_profile(
    color: [f64; 3],
    radius: [f64; 3],
    color_context: EeveeGBufferContext,
    radius_context: EeveeGBufferContext,
) -> ([f32; 3], [f32; 3]) {
    let color = closure_color_roundtrip(color.map(|value| value as f32), color_context);
    let radius = radius.map(|value| ior_pack(value as f32));
    let radius = closure_color_roundtrip(radius, radius_context).map(ior_unpack);
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

/// EEVEE's `AdditionalInfo` thickness roundtrip through its `UNORM_16_16` attachment.
pub(crate) fn eevee_gbuffer_thickness(thickness: Thickness) -> Thickness {
    let (value, slab) = match thickness {
        Thickness::None => (0.0_f32, true), // Source `Thickness::zero()` reports Slab.
        Thickness::Sphere(value) => (value as f32, false),
        Thickness::Slab(value) => (value as f32, true),
    };
    let packed = value / (1.0 + 2.0 * value);
    let packed = quantize_unorm(if slab { 1.0 - packed } else { packed }, 65_535);
    let value = if packed > 0.5 { 1.0 - packed } else { packed };
    let value = value / (1.0 - 2.0 * value);
    Thickness::new(f64::from(value), packed > 0.5)
}

/// EEVEE's G-buffer `closure_color_pack` roundtrip for a closure color stored at this data layer.
pub(crate) fn eevee_gbuffer_closure_color(color: DVec3, context: EeveeGBufferContext) -> DVec3 {
    DVec3::from_array(
        closure_color_roundtrip(color.to_array().map(|value| value as f32), context).map(f64::from),
    )
}

/// EEVEE's G-buffer roundtrip for a reflection or refraction's normalized roughness.
pub(crate) fn eevee_gbuffer_roughness(value: f64, context: EeveeGBufferContext) -> f64 {
    f64::from(eevee_gbuffer_unorm10(value as f32, 0, context))
}

/// EEVEE's G-buffer roundtrip for a refraction IOR stored in data component 1.
pub(crate) fn eevee_gbuffer_ior(value: f64, context: EeveeGBufferContext) -> f64 {
    f64::from(ior_unpack(eevee_gbuffer_unorm10(
        ior_pack(value as f32),
        1,
        context,
    )))
}

fn closure_color_roundtrip(color: [f32; 3], context: EeveeGBufferContext) -> [f32; 3] {
    let maximum = color[0].max(color[1]).max(color[2]);
    let exponent = if maximum > 4.0 {
        3
    } else if maximum > 2.0 {
        2
    } else {
        i32::from(maximum > 1.0)
    };
    let scale = 2.0_f32.powi(exponent);
    std::array::from_fn(|component| {
        eevee_gbuffer_unorm10(color[component] / scale, component, context) * scale
    })
}

/// EEVEE's dithered UNORM10 roundtrip for one G-buffer data component.
pub(crate) fn eevee_gbuffer_unorm10(
    value: f32,
    component: usize,
    context: EeveeGBufferContext,
) -> f32 {
    let noise = gbuffer_dither_noise(component, context);
    let step = 1.0_f32 / 1023.0;
    let value = if context.data_layer < 2 {
        (value + (noise - 0.5) * step).clamp(0.0, 1.0) * 1023.0
    } else {
        (value + noise * step).clamp(0.0, 1.0) * 1023.0
    };
    if context.data_layer < 2 {
        value.round() / 1023.0
    } else {
        value.floor() / 1023.0
    }
}

fn gbuffer_dither_noise(component: usize, context: EeveeGBufferContext) -> f32 {
    let seed = context.data_layer as f32 * 3.0 + component as f32;
    let pixel = [
        context.frag_coord[0] + seed * (47.0 * 0.695),
        context.frag_coord[1] + seed * (17.0 * 0.695),
    ];
    let noise = (0.067_110_56 * pixel[0] + 0.005_837_15 * pixel[1]).fract();
    (context.sample_offset[component] + 52.982_918 * noise).fract()
}

fn ior_pack(value: f32) -> f32 {
    const SCALE: f32 = 511.0 / 1023.0;
    const ONE: f32 = 512.0 / 1023.0;
    const BIAS: f32 = 510.0 / 1023.0;
    if value.to_bits() == 1.0_f32.to_bits() {
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
    if value.to_bits() == ONE.to_bits() {
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
            let stack = SubsurfaceResolveStack {
                samples,
                frame,
                sample_locations: &sample_locations,
                pixel_jitter,
                eye_start,
                eye_width,
                height,
                y,
                local_x,
            };
            for closure in center.sample.closures.iter().flatten().copied() {
                resolve_subsurface_closure(center, closure, &mut sample_color[pixel], stack);
            }
        }
    }
}

fn resolve_subsurface_closure(
    center: EeveeSubsurfacePixel,
    closure: EeveeSubsurfaceClosure,
    sample_color: &mut [f32; 4],
    stack: SubsurfaceResolveStack<'_>,
) {
    let SubsurfaceResolveStack {
        samples,
        frame,
        sample_locations,
        pixel_jitter,
        eye_start,
        eye_width,
        height,
        y,
        local_x,
    } = stack;
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
            * f64::from(radiance[channel])
            * center.sample.association_scale) as f32;
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
    // Match `precompute_samples_location`: Blender computes the angle in double, then stores float taps.
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    std::array::from_fn(|index| {
        let theta =
            (golden_angle * index as f64 + std::f64::consts::PI * 2.0 * f64::from(sample_u)) as f32;
        let radius = burley_sample(d, (sample_v + index as f32) / SSS_SAMPLE_COUNT as f32);
        let pdf_inverse = 1.0 / burley_pdf(d, radius);
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
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
        let offset = a - 0.8;
        let s = 1.9 - a + 3.5 * (offset * offset);
        let l = 0.25 * 0.318_309_87_f32 * radius[channel];
        l / s
    })
}

fn burley_sample(d: f32, x_rand: f32) -> f32 {
    let x = (f64::from(x_rand) * SSS_BURLEY_TRUNCATE_CDF) as f32;
    let mut radius = if f64::from(x) <= 0.9 {
        let x_squared = x * x;
        ((f64::from(x_squared) * 2.4).exp() - 1.0) as f32
    } else {
        15.0
    };
    for _ in 0..10 {
        let exp_r_3 = (-f64::from(radius) / 3.0).exp() as f32;
        let exp_r = exp_r_3 * exp_r_3 * exp_r_3;
        let f = (1.0 - 0.25 * f64::from(exp_r) - 0.75 * f64::from(exp_r_3) - f64::from(x)) as f32;
        let f_derivative = (0.25 * f64::from(exp_r) + 0.25 * f64::from(exp_r_3)) as f32;
        if f.abs() < 1.0e-6 || f_derivative == 0.0 {
            break;
        }
        radius = f64::from(radius - f / f_derivative).max(0.0) as f32;
    }
    radius * d
}

fn burley_pdf(d: f32, radius: f32) -> f32 {
    if radius >= SSS_BURLEY_TRUNCATE * d {
        return 0.0;
    }
    let exp_r_3_d = (-radius / (3.0 * d)).exp();
    let exp_r_d = exp_r_3_d * exp_r_3_d * exp_r_3_d;
    let burley_eval = (exp_r_d + exp_r_3_d) / (8.0 * PI * d);
    (f64::from(burley_eval) / SSS_BURLEY_TRUNCATE_CDF) as f32
}

/// GPU `burley_eval` retains the positive exponential tail; only the CPU proposal PDF is truncated.
fn burley_eval(d: [f32; 3], radius: f32) -> [f32; 3] {
    std::array::from_fn(|channel| {
        let exp_r_3_d = (-radius / (3.0 * d[channel])).exp();
        let exp_r_d = exp_r_3_d * exp_r_3_d * exp_r_3_d;
        (exp_r_d + exp_r_3_d) / (4.0 * d[channel])
    })
}

fn interleaved_gradient_noise(x: f32, y: f32) -> f32 {
    fract(52.982_918 * fract(0.067_110_56 * x + 0.005_837_15 * y))
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
    let exponent = ((value.to_bits() >> 23) & 0xff) as i32;
    if exponent - 127 + 15 >= 31 {
        return f32::INFINITY;
    }
    // SSS_RADIANCE is UFLOAT_11_11_10; reuse the existing measured truncating store conversion.
    super::eevee_transmission::truncate_ufloat(f64::from(value), mantissa_bits as i32) as f32
}
#[cfg(test)]
mod gbuffer_tests {
    use super::super::eevee_transmission::Thickness;
    use super::{
        EeveeGBufferContext, eevee_gbuffer_ior, eevee_gbuffer_normal, eevee_gbuffer_roughness,
        eevee_gbuffer_thickness, eevee_subsurface_gbuffer_profile,
    };
    use glam::DVec3;

    #[test]
    fn subsurface_profile_and_closure_parameters_match_dithered_gbuffer_storage() {
        let context = |data_layer| EeveeGBufferContext {
            frag_coord: [5.5, 7.5],
            sample_offset: [0.25, 0.5, 0.75],
            data_layer,
        };
        let (color, radius) = eevee_subsurface_gbuffer_profile(
            [0.72, 0.31, 0.12],
            [0.204, 0.0744, 0.0288],
            context(0),
            context(1),
        );
        let expected_color = [736.0_f32 / 1023.0, 317.0 / 1023.0, 123.0 / 1023.0];
        // Preserve the source f32 `ior_unpack(code / 1023)` operation order at the ROP store.
        let expected_radius = [
            104.0_f32 / 511.0,
            (38.0_f32 / 1023.0) / (511.0_f32 / 1023.0),
            15.0_f32 / 511.0,
        ];
        assert_eq!(color.map(f32::to_bits), expected_color.map(f32::to_bits));
        assert_eq!(radius.map(f32::to_bits), expected_radius.map(f32::to_bits));
        assert_eq!(
            eevee_gbuffer_roughness(0.26916, context(1)).to_bits(),
            f64::from(276.0_f32 / 1023.0).to_bits()
        );
        assert_eq!(
            eevee_gbuffer_roughness(0.26916, context(2)).to_bits(),
            f64::from(275.0_f32 / 1023.0).to_bits()
        );
        assert_eq!(
            eevee_gbuffer_ior(1.43, context(1)).to_bits(),
            1.432_584_166_526_794_4_f64.to_bits()
        );
    }

    #[test]
    fn shading_normal_matches_octahedral_unorm16_roundtrip() {
        let normal = eevee_gbuffer_normal(DVec3::new(0.06, 0.02, 0.998));
        let expected = DVec3::new(0.059_988_83, 0.019_985_31, 0.997_999);
        assert!(normal.distance(expected) < 2.0e-6);
    }

    #[test]
    fn thickness_matches_additional_info_unorm16_roundtrip() {
        // Blender stores the slab input as UNORM16 code 45133 before unpacking in f32.
        assert_eq!(
            eevee_gbuffer_thickness(Thickness::Slab(0.825)),
            Thickness::Slab(f64::from(f32::from_bits(1_062_416_476)))
        );
        assert_eq!(
            eevee_gbuffer_thickness(Thickness::Sphere(0.825)),
            Thickness::Sphere(f64::from(f32::from_bits(1_062_416_473)))
        );
        assert_eq!(
            eevee_gbuffer_thickness(Thickness::Sphere(1.0e-6)),
            Thickness::None
        );
        assert_eq!(
            eevee_gbuffer_thickness(Thickness::Slab(1.0e-6)),
            Thickness::None
        );
    }

    #[test]
    fn subsurface_radiance_uses_eevee_ufloat_truncation() {
        assert_eq!(
            super::quantize_radiance([0.6, 0.6, 0.3]),
            [0.59375, 0.59375, 0.296_875]
        );
        assert_eq!(
            super::quantize_radiance([0.436_523_44; 3]),
            [0.433_593_75, 0.433_593_75, 0.429_687_5]
        );
    }
    #[test]

    fn burley_gpu_profile_keeps_tail_while_cpu_pdf_truncates() {
        let d = 0.1;
        let radius = 16.0 * d;
        assert!(super::burley_eval([d; 3], radius)[0] > 0.0);
        assert_eq!(super::burley_pdf(d, radius), 0.0);
    }
}
