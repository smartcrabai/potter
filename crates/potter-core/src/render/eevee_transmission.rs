//! Transmission for realtime Beauty as EEVEE 5.2.2 evaluates it with its default material settings: no ray-traced
//! refraction (`Material.use_raytrace_refraction` off), so each refraction samples the light probe, and the object's
//! thickness (`Thickness`, `eevee_thickness_lib.bsl.hh`).
//!
//! EEVEE sources (Blender v5.2.2):
//! - `gpu/shaders/material/gpu_shader_material_principled.glsl`, the `MAT_REFRACTION` block: the transmission and
//!   reflection closures, with `bsdf_lut(F0, F90, sqrt(base), NV, roughness, ior, multiscatter)`.
//! - `draw/engines/eevee/shaders/eevee_nodetree_lib.bsl.hh`: `bsdf_lut` and `F_brdf_multi_scatter`.
//! - `eevee_bxdf_lut_lib.bsl.hh` and `eevee_utility_tx.bsl.hh`: the LUT coordinates and their filtering. The tables are
//!   `eevee_transmission_tables.rs`.
//! - `eevee_bxdf_microfacet.bsl.hh`: the dominant directions and the perceived transmission roughness.
//! - `eevee_closure.bsl.hh`, `bxdf_lightprobe_ray`: the probe direction, after `bxdf_ggx_context_amend_transmission`.
//! - `eevee_thickness_lib.bsl.hh`: the sphere and slab models of a thickness.
//! - `eevee_shader.cc`, `nodetree_thickness`: the thickness is the Material Output's Thickness socket when linked, and
//!   otherwise the smallest world-space bounds dimension.
//! - `eevee_deferred_combine.bsl.hh`: a refraction with a thickness takes its colour squared (two events).
//! - `eevee_lightprobe_sphere.bsl.hh` and the `UFLOAT_11_11_10` colour attachments of the capture (`eevee_shader.cc`):
//!   the probe radiance is truncated to that format, see [`probe_radiance`].
//!
//! The reflectance and transmittance come from EEVEE's split-sum LUT, so the Fresnel split and the roughness response
//! follow EEVEE's closures. Measured on Blender 5.2.2 with a white world of 1 and a white glass sphere (IOR 1.45,
//! roughness 0), EEVEE's centre pixel is 0.9668 and this model's R + T² is 0.9674.
//!
//! Direct light through glass is EEVEE's LTC light pass (`eevee_ltc.rs`): the reflection's and the transmission's colours
//! ([`GlassLight`]) integrate each light's emitter with the closure's inverse matrix. The transmission's shadow is the
//! thickness-amended ray of `shadow_eval` (`render/scene.rs`). Not reproduced: ray-traced refraction
//! (`use_raytrace_refraction`), which EEVEE leaves off by default.

use glam::DVec3;

use super::{
    eevee_ltc::{PackedMatrix, sample_matrix},
    eevee_transmission_tables::{brdf, bsdf, btdf},
};
use crate::shader::BsdfParams;

/// The last texel index of EEVEE's utility-texture LUTs, which are 64 texels across.
const LAST_TEXEL: usize = 63;
/// The number of roughness layers of the BSDF and BTDF LUTs, `UTIL_BSDF_LAYER_COUNT`.
const ROUGHNESS_LAYERS: usize = 16;

/// Linear blend of `a` and `b` at `t`.
pub(super) fn blend(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// EEVEE's `sample_lut` addressing for a coordinate in `[0, 1]`: the two texels around it and the blend weight of the
/// second. Texel `i` sits at coordinate `i / 63` (`UTIL_TEX_UV_SCALE` and `UTIL_TEX_UV_BIAS`).
pub(super) fn texel_weights(coordinate: f64) -> (usize, usize, f64) {
    let position = coordinate.clamp(0.0, 1.0) * LAST_TEXEL as f64;
    let low = (position.floor() as usize).min(LAST_TEXEL);
    let high = (low + 1).min(LAST_TEXEL);
    (low, high, position - low as f64)
}

/// The two roughness layers around `roughness` and the blend weight of the second (`sample_bsdf_lut`, layer
/// `16 · roughness`).
fn layer_weights(roughness: f64) -> (usize, usize, f64) {
    let layer = roughness.clamp(0.0, 1.0) * ROUGHNESS_LAYERS as f64;
    let low = (layer.floor() as usize).min(ROUGHNESS_LAYERS - 1);
    let high = (low + 1).min(ROUGHNESS_LAYERS - 1);
    (low, high, layer - low as f64)
}

/// `sqrt(1 - cos θ)`, the second LUT coordinate of the BRDF and IOR-above-one tables (`lut_coords_get`).
fn sqrt_one_minus(n_dot_v: f64) -> f64 {
    (1.0 - n_dot_v.clamp(0.0, 1.0)).sqrt()
}

/// One channel of EEVEE's GGX split sum (`GGXBrdfData`): channel 0 is the scale, channel 1 the bias.
fn brdf_channel(channel: usize, roughness: f64, n_dot_v: f64) -> f64 {
    let (x0, x1, fx) = texel_weights(roughness);
    let (y0, y1, fy) = texel_weights(sqrt_one_minus(n_dot_v));
    blend(
        blend(brdf(x0, y0, channel), brdf(x1, y0, channel), fx),
        blend(brdf(x0, y1, channel), brdf(x1, y1, channel), fx),
        fy,
    )
}

/// EEVEE's transmission factor for an IOR above one (`GGXBtdfGt1Data`) at roughness `roughness`, cosine `n_dot_v` and
/// `amplitude = (ior - 1) / (ior + 1)`. The roughness layers interpolate after the texel blend, as `sample_bsdf_lut`
/// does.
fn transmission_factor(roughness: f64, n_dot_v: f64, amplitude: f64) -> f64 {
    let (l0, l1, fl) = layer_weights(roughness);
    let (x0, x1, fx) = texel_weights(amplitude.sqrt());
    let (y0, y1, fy) = texel_weights(sqrt_one_minus(n_dot_v));
    let layer_value = |layer: usize| {
        blend(
            blend(btdf(layer, x0, y0), btdf(layer, x1, y0), fx),
            blend(btdf(layer, x0, y1), btdf(layer, x1, y1), fx),
            fy,
        )
    };
    blend(layer_value(l0), layer_value(l1), fl)
}

/// EEVEE's `GGXBsdfData` for an IOR below one: the scale, bias and transmission factor. Its coordinates are the
/// `GGXBsdfData::lut_coords_get` of `eevee_bxdf_lut_lib.bsl.hh`: `x = ior²`, and `y` remaps `cos θ` about the critical
/// angle `sqrt(1 - ior²)`.
fn bsdf_channels(ior: f64, roughness: f64, n_dot_v: f64) -> [f64; 3] {
    let critical_cos = (1.0 - ior * ior).max(0.0).sqrt();
    let offset = n_dot_v - critical_cos;
    let remapped = offset
        / if offset > 0.0 {
            1.0 - critical_cos
        } else {
            critical_cos
        };
    let x = (ior * ior).clamp(0.0, 1.0);
    let y = (remapped * 0.5 + 0.5).clamp(0.0, 1.0);
    let (l0, l1, fl) = layer_weights(roughness);
    let (x0, x1, fx) = texel_weights(x);
    let (y0, y1, fy) = texel_weights(y);
    let channel = |layer: usize, channel: usize| {
        blend(
            blend(
                bsdf(layer, x0, y0, channel),
                bsdf(layer, x1, y0, channel),
                fx,
            ),
            blend(
                bsdf(layer, x0, y1, channel),
                bsdf(layer, x1, y1, channel),
                fx,
            ),
            fy,
        )
    };
    [0, 1, 2].map(|index| blend(channel(l0, index), channel(l1, index), fl))
}

/// The reflectance and transmittance of a GGX glass closure, per channel, as EEVEE's `bsdf_lut` gives them with
/// multiscatter. `transmittance` includes the transmission tint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GlassSplit {
    pub(crate) reflectance: DVec3,
    pub(crate) transmittance: DVec3,
}

/// EEVEE's `bsdf_lut` for a glass closure. `f0` is the normal-incidence reflectance (EEVEE's `F0`, tinted by the
/// specular tint) and `transmission_tint` multiplies the transmittance.
pub(crate) fn glass_split(
    ior: f64,
    roughness: f64,
    n_dot_v: f64,
    f0: DVec3,
    transmission_tint: DVec3,
) -> GlassSplit {
    // EEVEE's `ior == 1`: no reflection, and the tint passes.
    if (ior - 1.0).abs() < 1.0e-9 {
        return GlassSplit {
            reflectance: DVec3::ZERO,
            transmittance: transmission_tint,
        };
    }
    let amplitude = (ior - 1.0) / (ior + 1.0);
    let real_f0 = amplitude * amplitude;
    let (scale, bias, transmission, f90) = if ior > 1.0 {
        // `bsdf_lut` replaces the default F90 of one with `saturate(2.33 / 0.33 · f0)`, which reaches one from an IOR of
        // about 1.33 up.
        (
            brdf_channel(0, roughness, n_dot_v),
            brdf_channel(1, roughness, n_dot_v),
            transmission_factor(roughness, n_dot_v, amplitude),
            (2.33 / 0.33 * amplitude).clamp(0.0, 1.0),
        )
    } else {
        let [scale, bias, transmission] = bsdf_channels(ior, roughness, n_dot_v);
        (scale, bias, transmission, 1.0)
    };

    let reflectance = f0 * scale + DVec3::splat(f90 * bias);
    let transmittance = (DVec3::ONE - f0) * transmission * transmission_tint;

    // `F_brdf_multi_scatter`'s normalisation: `Ess` is the energy the single-scatter terms keep, `Ems` the energy missing.
    let energy = real_f0 * scale + bias + (1.0 - real_f0) * transmission;
    let missing = 1.0 - energy;
    let average = if transmission_tint.max_element() <= 0.0 {
        f0 + (DVec3::splat(f90) - f0) / 21.0
    } else {
        transmission_tint
    };
    let normalisation = DVec3::ONE / (DVec3::ONE - average * missing).max(DVec3::splat(1.0e-6));
    GlassSplit {
        reflectance: reflectance * normalisation,
        transmittance: transmittance * normalisation,
    }
}

/// The mix weight of EEVEE's `bxdf_ggx_dominant_direction_*`: the probe direction leaves the normal for the specular
/// direction as the roughness grows.
fn dominant_mix(roughness: f64) -> f64 {
    let m = roughness * roughness;
    let smoothness = 1.0 - m;
    smoothness * (smoothness.sqrt() + m)
}

/// GLSL `reflect(incident, normal)`.
pub(crate) fn reflect(incident: DVec3, normal: DVec3) -> DVec3 {
    incident - normal * (2.0 * normal.dot(incident))
}

/// GLSL `refract(incident, normal, eta)`: zero under total internal reflection.
fn refract(incident: DVec3, normal: DVec3, eta: f64) -> DVec3 {
    let cos_incident = normal.dot(incident);
    let k = 1.0 - eta * eta * (1.0 - cos_incident * cos_incident);
    if k < 0.0 {
        DVec3::ZERO
    } else {
        incident * eta - normal * (eta * cos_incident + k.sqrt())
    }
}

/// EEVEE's `bxdf_ggx_dominant_direction_reflection` for the reflection of `view` at `normal`.
pub(crate) fn reflection_direction(normal: DVec3, view: DVec3, roughness: f64) -> DVec3 {
    // EEVEE's `-reflect(V, N)`, which is the reflection of the incident direction `-V`.
    let mirror = reflect(-view, normal);
    normal
        .lerp(mirror, dominant_mix(roughness))
        .normalize_or(normal)
}

/// EEVEE's `bxdf_ggx_dominant_direction_transmission`: `ior` is the argument EEVEE passes, whose reciprocal refracts
/// `-view`, mixed toward `-normal` by the roughness.
fn dominant_transmission(normal: DVec3, view: DVec3, ior: f64, roughness: f64) -> DVec3 {
    let refracted = refract(-view, normal, 1.0 / ior);
    (-normal)
        .lerp(refracted, dominant_mix(roughness))
        .normalize_or(-normal)
}

/// EEVEE's `bxdf_ggx_perceived_roughness_transmission`: the roughness the environment is blurred with for a refraction
/// at `ior`.
pub(crate) fn transmission_roughness(roughness: f64, ior: f64) -> f64 {
    let factor = if ior > 1.0 {
        1.0 - 1.0 / ior
    } else {
        (1.0 - ior).clamp(0.0, 1.0) * 0.64
    };
    roughness * factor.max(0.0).sqrt()
}

/// `ThicknessIsect::hit_N` of the sphere model (`eevee_thickness_lib.bsl.hh`): the normal where the entry direction
/// `entry` leaves a sphere that is tangent to the shading point with normal `normal`.
pub(crate) fn sphere_exit_normal(normal: DVec3, entry: DVec3) -> DVec3 {
    let cos_alpha = entry.dot(-normal);
    (normal + entry * (2.0 * cos_alpha)).normalize_or(normal)
}

/// An object's refraction thickness, as EEVEE's `Thickness` holds it: a value and a model. A zero value is no thickness.
/// The probe samples only the exit normal, which does not depend on the size, so the value matters only as zero or not;
/// it places the exit point, which the probe does not read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Thickness {
    None,
    /// A sphere of this diameter tangent to the shading point (`ThicknessMode::Sphere`, the default).
    Sphere(f64),
    /// A plane at this distance behind the shading point (`ThicknessMode::Slab`).
    Slab(f64),
}

impl Thickness {
    /// The thickness value (`Thickness::value`): zero for no thickness, whichever model is set.
    pub(crate) fn value(self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Sphere(value) | Self::Slab(value) => value,
        }
    }

    /// EEVEE's `Thickness::from(value, mode)`: no thickness for a zero value, otherwise the model's.
    pub(crate) fn new(value: f64, slab: bool) -> Self {
        if value <= 0.0 {
            Self::None
        } else if slab {
            Self::Slab(value)
        } else {
            Self::Sphere(value)
        }
    }
}

/// The direction EEVEE samples the environment at for a refraction of `view` at `normal` into `ior`.
///
/// Without a thickness this is the refracted dominant direction. With one, EEVEE first amends the shading point with the
/// thickness model (`bxdf_ggx_context_amend_transmission`), and the probe takes the direction that leaves it. A sphere
/// crosses at `L · cos α · diameter`, where `L` is the entry direction, and its exit normal is `normalize(N + 2 cos α · L)`.
/// A slab's exit normal is `-N` at its plane. Both then refract `L` out through the amended normal.
pub(crate) fn transmission_direction(
    normal: DVec3,
    view: DVec3,
    ior: f64,
    roughness: f64,
    thickness: Thickness,
) -> DVec3 {
    let perceived = transmission_roughness(roughness, ior);
    let entry = dominant_transmission(normal, view, ior, perceived);
    let (amended_normal, amended_view) = match thickness {
        Thickness::None => return entry,
        Thickness::Sphere(_) => (-sphere_exit_normal(normal, entry), -entry),
        Thickness::Slab(_) => (normal, -entry),
    };
    // The probe's amended normal and view, with EEVEE's `1 / ior` argument (a thickness inverts it).
    dominant_transmission(amended_normal, amended_view, 1.0 / ior, perceived)
}

/// The radiance a light probe holds: EEVEE captures the world into `UFLOAT_11_11_10` colour attachments, which truncate
/// the mantissa to 6 bits for red and green and 5 for blue (no rounding). Measured on Blender 5.2.2, a world colour of
/// 0.6 reads back through a glass sphere as 0.59375, and 0.3 in blue as 0.296875.
pub(crate) fn probe_radiance(radiance: DVec3) -> DVec3 {
    DVec3::new(
        truncate_ufloat(radiance.x, 6),
        truncate_ufloat(radiance.y, 6),
        truncate_ufloat(radiance.z, 5),
    )
}

/// The irradiance a light probe stores for diffuse light: EEVEE's spherical-harmonic irradiance of a uniform world
/// reads `1 - 2^-17` of the radiance before capture (a float32 round trip), so the captured value is truncated just
/// below the radiance. Measured on Blender 5.2.2: a world of 1 reads 0.9921875 (blue 0.984375) in a white diffuse, 0.5
/// reads 0.4960938, 0.25 reads 0.2480469 and 0.75 reads 0.7421875. A world of 0.6 reads 0.59375, as the glossy does.
pub(crate) fn irradiance_probe_radiance(radiance: DVec3) -> DVec3 {
    probe_radiance(radiance * (1.0 - 1.0 / 131_072.0))
}

/// One channel of a `UFLOAT` with `mantissa_bits` mantissa bits, a 5-bit exponent (bias 15) and truncation.
fn truncate_ufloat(value: f64, mantissa_bits: i32) -> f64 {
    if value.is_nan() || value <= 0.0 {
        return 0.0;
    }
    let largest = (2.0 - 2.0_f64.powi(-mantissa_bits)) * 32_768.0;
    let value = value.min(largest);
    let exponent = value.log2().floor().clamp(-14.0, 15.0);
    let step = (exponent - f64::from(mantissa_bits)).exp2();
    (value / step).floor() * step
}

/// The glass closures of one shading point, as EEVEE weights them, with the directions to sample the environment at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GlassClosures {
    /// The reflection's weight times its colour: the environment at `reflection_direction` is multiplied by it.
    pub(crate) reflection: DVec3,
    pub(crate) reflection_direction: DVec3,
    pub(crate) reflection_roughness: f64,
    /// The transmission's weight times its colour, squared when the object has a thickness (two events).
    pub(crate) transmission: DVec3,
    pub(crate) transmission_direction: DVec3,
    pub(crate) transmission_roughness: f64,
}

/// The direct light of the glass closures at one shading point, as EEVEE's light pass evaluates them
/// (`eevee_light_eval.bsl.hh`, `eval_single_closure`): each closure's colour, with the inverse LTC matrix and the normal
/// its lobe integrates around.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GlassLight {
    /// The reflection closure's colour: its weight times the glass reflectance and the tint, over the layers above.
    pub(crate) reflection: DVec3,
    /// The reflection's inverse LTC matrix, sampled at `N · V` and the roughness (`bxdf_ggx_light_reflection`).
    pub(crate) reflection_matrix: PackedMatrix,
    /// The transmission closure's colour, squared with a thickness (two events; `eevee_deferred_combine.bsl.hh`).
    pub(crate) transmission: DVec3,
    /// The normal the transmission lobe integrates around (`light.N`): `-N`, or the exit normal of a thickness.
    pub(crate) transmission_normal: DVec3,
    /// The transmission's inverse LTC matrix, sampled at the refracted cosine and the perceived roughness.
    pub(crate) transmission_matrix: PackedMatrix,
}

/// The inputs both glass evaluations share at a shading point: the clamped IOR and roughness, the `bsdf_lut` split, the
/// reflection tint, and the lobe weight `lower · (1 - metallic) · transmission`.
struct GlassLobe {
    ior: f64,
    roughness: f64,
    split: GlassSplit,
    reflection_tint: DVec3,
    lobe_weight: DVec3,
}

/// The glass lobe at a shading point, or `None` when its weight `(1 - metallic) · transmission` is zero or the view is
/// below the normal.
fn glass_lobe(params: &BsdfParams, normal: DVec3, view: DVec3, lower: DVec3) -> Option<GlassLobe> {
    let weight = (1.0 - params.metallic.clamp(0.0, 1.0)) * params.transmission.clamp(0.0, 1.0);
    if weight <= 0.0 || normal.dot(view) <= 0.0 {
        return None;
    }
    let ior = params.ior.max(1.0e-5);
    let roughness = params.roughness.clamp(0.0, 1.0);
    let amplitude = (ior - 1.0) / (ior + 1.0);
    // EEVEE's `F0 = F0_from_ior(ior) * reflection_tint` for a Principled, where the Glass node's Color is the tint.
    let f0 = (DVec3::from_array(params.specular_tint).max(DVec3::ZERO) * amplitude * amplitude)
        .clamp(DVec3::ZERO, DVec3::ONE);
    let transmission_tint = DVec3::from_array(params.glass_transmission_tint).max(DVec3::ZERO);
    let reflection_tint = DVec3::from_array(params.glass_reflection_tint).max(DVec3::ZERO);
    Some(GlassLobe {
        ior,
        roughness,
        split: glass_split(ior, roughness, normal.dot(view), f0, transmission_tint),
        reflection_tint,
        lobe_weight: lower * weight,
    })
}

/// The glass closures of a Principled Transmission or Glass BSDF at a shading point with `normal`, seen along `view`,
/// with `lower` the layers above it (coat and sheen) and `thickness` the object's thickness. `None` when the
/// transmission weight `(1 - metallic) · transmission` is zero or the view is below the normal.
pub(crate) fn glass_closures(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    lower: DVec3,
    thickness: Thickness,
) -> Option<GlassClosures> {
    let lobe = glass_lobe(params, normal, view, lower)?;
    let reflection = lobe.split.reflectance * lobe.reflection_tint * lobe.lobe_weight;
    let mut transmission = lobe.split.transmittance * lobe.lobe_weight;
    if thickness != Thickness::None {
        transmission *= transmission;
    }
    Some(GlassClosures {
        reflection,
        reflection_direction: reflection_direction(normal, view, lobe.roughness),
        reflection_roughness: lobe.roughness,
        transmission,
        transmission_direction: transmission_direction(
            normal,
            view,
            lobe.ior,
            lobe.roughness,
            thickness,
        ),
        transmission_roughness: transmission_roughness(lobe.roughness, lobe.ior),
    })
}

/// The direct light of the glass closures at a shading point: `glass_closures`' colours with the LTC inputs of
/// `bxdf_ggx_light_reflection` and `bxdf_ggx_light_transmission`. `None` where `glass_closures` is `None`.
pub(crate) fn glass_light(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    lower: DVec3,
    thickness: Thickness,
) -> Option<GlassLight> {
    let lobe = glass_lobe(params, normal, view, lower)?;
    let mut transmission = lobe.split.transmittance * lobe.lobe_weight;
    if thickness != Thickness::None {
        transmission *= transmission;
    }
    let (transmission_normal, cos_theta) =
        transmission_light_normal(normal, view, lobe.ior, lobe.roughness, thickness);
    Some(GlassLight {
        reflection: lobe.split.reflectance * lobe.reflection_tint * lobe.lobe_weight,
        reflection_matrix: sample_matrix(normal.dot(view), lobe.roughness),
        transmission,
        transmission_normal,
        transmission_matrix: sample_matrix(
            cos_theta,
            transmission_roughness(lobe.roughness, lobe.ior),
        ),
    })
}

/// `bxdf_ggx_light_transmission`: the normal the transmission lobe integrates around, and the cosine its matrix is
/// sampled at. Without a thickness the lobe is `refract(-V, N, 1 / ior)` about `-N`. With one, the point moves to the
/// sphere's exit (or the slab's plane, which is `-N`) along the entry direction, and the cosine is that of the exit's
/// refraction `refract(entry, -exit, ior)` about `-exit`.
fn transmission_light_normal(
    normal: DVec3,
    view: DVec3,
    ior: f64,
    roughness: f64,
    thickness: Thickness,
) -> (DVec3, f64) {
    match thickness {
        Thickness::None => {
            let refracted = refract(-view, normal, 1.0 / ior);
            (-normal, (-normal).dot(refracted))
        }
        Thickness::Sphere(_) | Thickness::Slab(_) => {
            let perceived = transmission_roughness(roughness, ior);
            let entry = dominant_transmission(normal, view, ior, perceived);
            let exit = match thickness {
                Thickness::Sphere(_) => sphere_exit_normal(normal, entry),
                _ => -normal,
            };
            let refracted = refract(entry, -exit, ior);
            (exit, exit.dot(refracted))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Material;

    /// The white glass sphere of the EEVEE probe: Principled, IOR 1.45, Transmission 1, at `roughness`.
    fn glass(roughness: f64) -> BsdfParams {
        BsdfParams::from_material(
            &Material {
                transmission: 1.0,
                ior: 1.45,
                roughness,
                base_color: [1.0, 1.0, 1.0, 1.0],
                ..Material::default()
            },
            DVec3::Z,
        )
    }

    /// The closures of `params` looking straight down the normal, with `thickness`.
    fn normal_view(params: &BsdfParams, thickness: Thickness) -> Result<GlassClosures, String> {
        glass_closures(params, DVec3::Z, DVec3::Z, DVec3::ONE, thickness)
            .ok_or_else(|| "a view along the normal has a glass lobe".to_owned())
    }

    #[test]
    fn uniform_world_reflection_plus_squared_transmission_matches_eevee_at_normal_incidence()
    -> Result<(), String> {
        // Blender 5.2.2 EEVEE, white world of 1, Principled white, roughness 0, IOR 1.45: centre pixel 0.9668.
        let closures = normal_view(&glass(0.0), Thickness::Sphere(2.0))?;
        let total = closures.reflection + closures.transmission;
        for channel in total.to_array() {
            if (channel - 0.9668).abs() >= 0.002 {
                return Err(format!(
                    "uniform-world glass is {channel}, EEVEE's centre is 0.9668"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn a_thin_object_transmits_once_and_a_thick_one_twice() -> Result<(), String> {
        let params = glass(0.0);
        let once = normal_view(&params, Thickness::None)?;
        let twice = normal_view(&params, Thickness::Sphere(2.0))?;
        if (once.transmission.x * once.transmission.x - twice.transmission.x).abs() >= 1.0e-9 {
            return Err("the thick transmission is not the square of the single event".to_owned());
        }
        Ok(())
    }

    #[test]
    fn transmission_tint_is_squared_with_the_thickness() -> Result<(), String> {
        let mut params = glass(0.0);
        params.glass_transmission_tint = [0.5, 0.5, 0.5];
        let once = normal_view(&params, Thickness::None)?;
        let twice = normal_view(&params, Thickness::Sphere(2.0))?;
        if (twice.transmission.x - once.transmission.x * once.transmission.x).abs() >= 1.0e-9 {
            return Err(
                "the tinted thick transmission is not the square of the single event".to_owned(),
            );
        }
        Ok(())
    }

    #[test]
    fn straight_through_probe_at_normal_incidence_for_every_thickness_model() {
        let view = DVec3::Z;
        let normal = DVec3::Z;
        for thickness in [
            Thickness::None,
            Thickness::Sphere(2.0),
            Thickness::Slab(2.0),
        ] {
            assert!(
                (transmission_direction(normal, view, 1.45, 0.0, thickness) + view).length()
                    < 1.0e-9,
                "{thickness:?} keeps the straight line at normal incidence"
            );
        }
    }

    #[test]
    fn a_zero_thickness_value_is_no_thickness() {
        assert_eq!(Thickness::new(0.0, false), Thickness::None);
        assert_eq!(Thickness::new(0.5, true), Thickness::Slab(0.5));
        assert_eq!(Thickness::new(0.5, false), Thickness::Sphere(0.5));
    }

    #[test]
    fn perceived_transmission_roughness_follows_eevee() {
        // roughness * sqrt(1 - 1 / ior) for an IOR above one.
        assert!((transmission_roughness(1.0, 1.5) - (1.0_f64 / 3.0).sqrt()).abs() < 1.0e-12);
        assert!(transmission_roughness(0.0, 1.5).abs() < 1.0e-12);
    }

    #[test]
    fn smooth_mirror_reflects_the_view_about_the_normal() {
        // The view leans 45 degrees toward +X, so the mirror direction leans toward -X.
        let normal = DVec3::Z;
        let view = DVec3::new(1.0, 0.0, 1.0).normalize();
        let reflected = reflection_direction(normal, view, 0.0);
        assert!((reflected - DVec3::new(-1.0, 0.0, 1.0).normalize()).length() < 1.0e-9);
    }

    #[test]
    fn brdf_split_sum_is_one_for_a_smooth_surface_facing_the_viewer() {
        // EEVEE's table at roughness 0 and cos 1: scale 1, bias 0.
        assert!((brdf_channel(0, 0.0, 1.0) - 1.0).abs() < 1.0e-5);
        assert!(brdf_channel(1, 0.0, 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn ior_of_one_passes_its_tint() -> Result<(), String> {
        let pass = glass_split(1.0, 0.0, 1.0, DVec3::splat(0.04), DVec3::splat(0.5));
        if pass.reflectance != DVec3::ZERO || pass.transmittance != DVec3::splat(0.5) {
            return Err("an IOR of one does not pass its tint".to_owned());
        }
        Ok(())
    }

    #[test]
    fn ior_below_one_has_finite_reflection_and_transmission() -> Result<(), String> {
        // Smooth glass at IOR 0.8 seen along the normal: a small Fresnel reflection and most of the light transmitted.
        let split = glass_split(0.8, 0.0, 1.0, DVec3::splat(0.0144), DVec3::ONE);
        let total = split.reflectance + split.transmittance;
        for channel in total.to_array() {
            if !channel.is_finite() || !(0.5..=1.05).contains(&channel) {
                return Err(format!(
                    "IOR 0.8 glass sums to {channel}, expected about one"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn diffuse_irradiance_capture_matches_blender_measurements() -> Result<(), String> {
        // Blender 5.2.2 white diffuse under a uniform world: 1 -> (0.9921875, 0.9921875, 0.984375), 0.5 -> 0.4960938
        // in red and green, 0.25 -> 0.2480469, 0.75 -> 0.7421875; the glossy read of 0.6 is 0.59375.
        let one = irradiance_probe_radiance(DVec3::ONE);
        if (one.x - 0.992_187_5).abs() > 1.0e-12 || (one.z - 0.984_375).abs() > 1.0e-12 {
            return Err(format!("a unit world reads {one:?} in the diffuse"));
        }
        let half = irradiance_probe_radiance(DVec3::splat(0.5));
        if (half.x - 0.496_093_75).abs() > 1.0e-12 {
            return Err(format!("a 0.5 world reads {half:?} in the diffuse"));
        }
        let quarter = irradiance_probe_radiance(DVec3::splat(0.25)).x;
        let three_quarters = irradiance_probe_radiance(DVec3::splat(0.75)).x;
        if (quarter - 0.248_046_875).abs() > 1.0e-12
            || (three_quarters - 0.742_187_5).abs() > 1.0e-12
        {
            return Err(format!(
                "quarter and three-quarter worlds read {quarter} and {three_quarters}"
            ));
        }
        if (probe_radiance(DVec3::splat(0.6)).x - 0.59375).abs() > 1.0e-12 {
            return Err("the glossy read of 0.6 is not 0.59375".to_owned());
        }
        Ok(())
    }

    #[test]
    fn probe_radiance_truncates_to_the_capture_format() -> Result<(), String> {
        // 0.6 = 0.5 · 1.2 has a 6-bit mantissa of 76/64 · 0.5 → 0.59375; 0.3 in blue, 5 bits → 0.296875.
        let probe = probe_radiance(DVec3::new(0.6, 0.6, 0.3));
        if (probe.x - 0.59375).abs() > 1.0e-12 || (probe.z - 0.296_875).abs() > 1.0e-12 {
            return Err(format!(
                "probe radiance {probe:?} is not truncated to UFLOAT_11_11_10"
            ));
        }
        if probe_radiance(DVec3::ONE) != DVec3::ONE {
            return Err("1.0 is not exact in the capture format".to_owned());
        }
        Ok(())
    }
}
