//! Transmission for realtime Beauty as EEVEE 5.2.2 evaluates it with its default material settings: no ray-traced
//! refraction (`Material.use_raytrace_refraction` off), so each refraction samples the light probe, and the
//! object's sphere thickness (`thickness_mode` Sphere, `Material.refraction_depth` unused).
//!
//! EEVEE sources (Blender v5.2.2):
//! - `gpu/shaders/material/gpu_shader_material_principled.glsl`, the `MAT_REFRACTION` block: the transmission and
//!   reflection closures, with `bsdf_lut(F0, F90, sqrt(base), NV, roughness, ior, multiscatter)`.
//! - `draw/engines/eevee/shaders/eevee_nodetree_lib.bsl.hh`: `bsdf_lut` (IOR above one) and `F_brdf_multi_scatter`.
//! - `eevee_bxdf_lut_lib.bsl.hh` and `eevee_utility_tx.bsl.hh`: the LUT coordinates and their filtering. The tables
//!   are `eevee_transmission_tables.rs`.
//! - `eevee_bxdf_microfacet.bsl.hh`: the dominant directions and the perceived transmission roughness.
//! - `eevee_closure.bsl.hh`, `bxdf_lightprobe_ray`: the probe direction, after `bxdf_ggx_context_amend_transmission`.
//! - `eevee_thickness_lib.bsl.hh`: the sphere model, which crosses a sphere of the object's thickness.
//! - `eevee_shader.cc`, `nodetree_thickness`: the default thickness is the smallest world-space bounds dimension.
//! - `eevee_deferred_combine.bsl.hh`: a refraction with a thickness takes its colour squared (two events).
//!
//! The reflectance and transmittance come from EEVEE's split-sum LUT, so the Fresnel split and the roughness response
//! follow EEVEE's closures rather than an analytic approximation. Measured on Blender 5.2.2 with a uniform world of
//! 1 and a white glass sphere (IOR 1.45, roughness 0), EEVEE's centre pixel is 0.9668, this model's R + T² is 0.9674.
//!
//! Not reproduced (reported as losses): IOR below one, which EEVEE reads from its `bsdf_ggx` table (not embedded, so
//! such a closure has no environment term here), and the Thickness socket and thickness-from-shadow amendments, which
//! replace the bounds-derived thickness.

use glam::DVec3;

use super::eevee_transmission_tables::{BRDF_SCALE_BIAS, BTDF_GT_ONE};
use crate::shader::BsdfParams;

/// The last texel index of EEVEE's utility-texture LUTs, which are 64 texels across.
const LAST_TEXEL: usize = 63;
/// The number of roughness layers of the BSDF LUT, `UTIL_BSDF_LAYER_COUNT`.
const ROUGHNESS_LAYERS: usize = 16;

/// Linear blend of `a` and `b` at `t`.
fn blend(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// EEVEE's `sample_lut` addressing for a coordinate in `[0, 1]`: the two texels around it and the blend weight of the
/// second. Texel `i` sits at coordinate `i / 63` (`UTIL_TEX_UV_SCALE` and `UTIL_TEX_UV_BIAS`).
fn texel_weights(coordinate: f64) -> (usize, usize, f64) {
    let position = coordinate.clamp(0.0, 1.0) * LAST_TEXEL as f64;
    let low = (position.floor() as usize).min(LAST_TEXEL);
    let high = (low + 1).min(LAST_TEXEL);
    (low, high, position - low as f64)
}

/// `sqrt(1 - cos θ)`, the second LUT coordinate of every table (`lut_coords_get`).
fn sqrt_one_minus(n_dot_v: f64) -> f64 {
    (1.0 - n_dot_v.clamp(0.0, 1.0)).sqrt()
}

/// One channel of EEVEE's GGX split sum (`GGXBrdfData`): channel 0 is the scale, channel 1 the bias.
fn brdf_channel(channel: usize, roughness: f64, n_dot_v: f64) -> f64 {
    let (x0, x1, fx) = texel_weights(roughness);
    let (y0, y1, fy) = texel_weights(sqrt_one_minus(n_dot_v));
    let texel = |y: usize, x: usize| f64::from(BRDF_SCALE_BIAS[y][x][channel]);
    blend(
        blend(texel(y0, x0), texel(y0, x1), fx),
        blend(texel(y1, x0), texel(y1, x1), fx),
        fy,
    )
}

/// EEVEE's transmission factor for an IOR above one (`GGXBtdfGt1Data`) at roughness `roughness`, cosine `n_dot_v`
/// and `amplitude = (ior - 1) / (ior + 1)`. The roughness layers interpolate after the texel blend, as
/// `sample_bsdf_lut` does.
fn transmission_factor(roughness: f64, n_dot_v: f64, amplitude: f64) -> f64 {
    let layer = roughness.clamp(0.0, 1.0) * ROUGHNESS_LAYERS as f64;
    let low_layer = (layer.floor() as usize).min(ROUGHNESS_LAYERS - 1);
    let high_layer = (low_layer + 1).min(ROUGHNESS_LAYERS - 1);
    let layer_weight = layer - low_layer as f64;
    let (x0, x1, fx) = texel_weights(amplitude.sqrt());
    let (y0, y1, fy) = texel_weights(sqrt_one_minus(n_dot_v));
    let layer_value = |index: usize| {
        let texel = |y: usize, x: usize| f64::from(BTDF_GT_ONE[index][y][x]);
        blend(
            blend(texel(y0, x0), texel(y0, x1), fx),
            blend(texel(y1, x0), texel(y1, x1), fx),
            fy,
        )
    };
    blend(
        layer_value(low_layer),
        layer_value(high_layer),
        layer_weight,
    )
}

/// The reflectance and transmittance of a GGX glass closure, per channel, as EEVEE's `bsdf_lut` gives them for an
/// IOR above one with multiscatter. `transmittance` includes the transmission tint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GlassSplit {
    pub(crate) reflectance: DVec3,
    pub(crate) transmittance: DVec3,
}

/// EEVEE's `bsdf_lut` for a glass closure. `f0` is the normal-incidence reflectance (EEVEE's `F0`, tinted by the
/// specular tint), and `transmission_tint` multiplies the transmittance. `None` for an IOR below one, whose table
/// EEVEE reads from `bsdf_ggx`, which Potter does not embed.
pub(crate) fn glass_split(
    ior: f64,
    roughness: f64,
    n_dot_v: f64,
    f0: DVec3,
    transmission_tint: DVec3,
) -> Option<GlassSplit> {
    if ior < 1.0 {
        return None;
    }
    // EEVEE's `ior == 1`: no reflection, and the tint passes.
    if (ior - 1.0).abs() < 1.0e-9 {
        return Some(GlassSplit {
            reflectance: DVec3::ZERO,
            transmittance: transmission_tint,
        });
    }
    let amplitude = (ior - 1.0) / (ior + 1.0);
    let real_f0 = amplitude * amplitude;
    // `bsdf_lut` replaces the default F90 of one with `saturate(2.33 / 0.33 · f0)`, which reaches one from an IOR of about 1.33.
    let f90 = (2.33 / 0.33 * amplitude).clamp(0.0, 1.0);
    let scale = brdf_channel(0, roughness, n_dot_v);
    let bias = brdf_channel(1, roughness, n_dot_v);
    let transmission = transmission_factor(roughness, n_dot_v, amplitude);

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
    Some(GlassSplit {
        reflectance: reflectance * normalisation,
        transmittance: transmittance * normalisation,
    })
}

/// The mix weight of EEVEE's `bxdf_ggx_dominant_direction_*`: the probe direction leaves the normal for the specular
/// direction as the roughness grows.
fn dominant_mix(roughness: f64) -> f64 {
    let m = roughness * roughness;
    let smoothness = 1.0 - m;
    smoothness * (smoothness.sqrt() + m)
}

/// GLSL `reflect(incident, normal)`.
fn reflect(incident: DVec3, normal: DVec3) -> DVec3 {
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

/// EEVEE's `bxdf_ggx_dominant_direction_transmission`: `ior` is the argument EEVEE passes, whose reciprocal
/// refracts `-view`, mixed toward `-normal` by the roughness.
fn dominant_transmission(normal: DVec3, view: DVec3, ior: f64, roughness: f64) -> DVec3 {
    let refracted = refract(-view, normal, 1.0 / ior);
    (-normal)
        .lerp(refracted, dominant_mix(roughness))
        .normalize_or(-normal)
}

/// EEVEE's `bxdf_ggx_perceived_roughness_transmission`: the roughness the environment is blurred with for a
/// refraction at `ior`.
pub(crate) fn transmission_roughness(roughness: f64, ior: f64) -> f64 {
    let factor = if ior > 1.0 {
        1.0 - 1.0 / ior
    } else {
        (1.0 - ior).clamp(0.0, 1.0) * 0.64
    };
    roughness * factor.max(0.0).sqrt()
}

/// The direction EEVEE samples the environment at for a refraction of `view` at `normal` into `ior`.
///
/// Without a thickness this is the refracted dominant direction. With one, EEVEE first crosses the sphere model of
/// `bxdf_ggx_context_amend_transmission`: the entry direction `L` meets the sphere at `L · cos α · thickness`, where the
/// exit normal is `normalize(N + 2 cos α · L)`, and the probe takes the direction leaving the sphere there.
pub(crate) fn transmission_direction(
    normal: DVec3,
    view: DVec3,
    ior: f64,
    roughness: f64,
    thickness: f64,
) -> DVec3 {
    let perceived = transmission_roughness(roughness, ior);
    let entry = dominant_transmission(normal, view, ior, perceived);
    if thickness <= 0.0 {
        return entry;
    }
    let cos_alpha = entry.dot(-normal);
    let exit_normal = (normal + entry * (2.0 * cos_alpha)).normalize_or(normal);
    // The probe's amended normal is `-exit_normal` and its view `-entry`, with EEVEE's `1 / ior` argument.
    dominant_transmission(-exit_normal, -entry, 1.0 / ior, perceived)
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

/// The glass closures of a Principled Transmission or Glass BSDF at a shading point with `normal`, seen along
/// `view`, with `lower` the layers above it (coat and sheen) and `thickness` the object's sphere thickness (zero for
/// none). `None` when the transmission weight `(1 - metallic) · transmission` is zero, the view is below the normal, or
/// the IOR is below one.
pub(crate) fn glass_closures(
    params: &BsdfParams,
    normal: DVec3,
    view: DVec3,
    lower: DVec3,
    thickness: f64,
) -> Option<GlassClosures> {
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
    let split = glass_split(ior, roughness, normal.dot(view), f0, transmission_tint)?;

    let lobe_weight = lower * weight;
    let reflection = split.reflectance * reflection_tint * lobe_weight;
    let mut transmission = split.transmittance * lobe_weight;
    if thickness > 0.0 {
        transmission *= transmission;
    }
    Some(GlassClosures {
        reflection,
        reflection_direction: reflection_direction(normal, view, roughness),
        reflection_roughness: roughness,
        transmission,
        transmission_direction: transmission_direction(normal, view, ior, roughness, thickness),
        transmission_roughness: transmission_roughness(roughness, ior),
    })
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
    fn normal_view(params: &BsdfParams, thickness: f64) -> Result<GlassClosures, String> {
        glass_closures(params, DVec3::Z, DVec3::Z, DVec3::ONE, thickness)
            .ok_or_else(|| "a view along the normal has a glass lobe".to_owned())
    }

    #[test]
    fn uniform_world_reflection_plus_squared_transmission_matches_eevee_at_normal_incidence()
    -> Result<(), String> {
        // Blender 5.2.2 EEVEE, uniform world of 1, Principled white, roughness 0, IOR 1.45: centre pixel 0.9668.
        let closures = normal_view(&glass(0.0), 2.0)?;
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
        let once = normal_view(&params, 0.0)?;
        let twice = normal_view(&params, 2.0)?;
        if (once.transmission.x * once.transmission.x - twice.transmission.x).abs() >= 1.0e-9 {
            return Err("the thick transmission is not the square of the single event".to_owned());
        }
        Ok(())
    }

    #[test]
    fn transmission_tint_is_squared_with_the_thickness() -> Result<(), String> {
        let mut params = glass(0.0);
        params.glass_transmission_tint = [0.5, 0.5, 0.5];
        let once = normal_view(&params, 0.0)?;
        let twice = normal_view(&params, 2.0)?;
        if (twice.transmission.x - once.transmission.x * once.transmission.x).abs() >= 1.0e-9 {
            return Err(
                "the tinted thick transmission is not the square of the single event".to_owned(),
            );
        }
        Ok(())
    }

    #[test]
    fn straight_through_probe_at_normal_incidence_with_and_without_thickness() {
        let view = DVec3::Z;
        let normal = DVec3::Z;
        assert!(
            (transmission_direction(normal, view, 1.45, 0.0, 0.0) + view).length() < 1.0e-9,
            "without thickness the probe samples straight through"
        );
        assert!(
            (transmission_direction(normal, view, 1.45, 0.0, 2.0) + view).length() < 1.0e-9,
            "a sphere of thickness keeps the straight line at normal incidence"
        );
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
    fn ior_below_one_has_no_split() -> Result<(), String> {
        let f0 = DVec3::splat(0.04);
        if glass_split(0.8, 0.0, 1.0, f0, DVec3::ONE).is_some() {
            return Err("an IOR below one has a table in Potter".to_owned());
        }
        let pass = glass_split(1.0, 0.0, 1.0, f0, DVec3::splat(0.5))
            .ok_or_else(|| "an IOR of one has no split".to_owned())?;
        if pass.reflectance != DVec3::ZERO || pass.transmittance != DVec3::splat(0.5) {
            return Err("an IOR of one does not pass its tint".to_owned());
        }
        Ok(())
    }
}
