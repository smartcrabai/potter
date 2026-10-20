//! Blender 5.2.2 Workbench STUDIO lighting, which the raster Solid preview shades with.
//!
//! Workbench lights a surface in view space from the four lights of its studio light. Potter
//! models the factory `Default` studio light, the one `BKE_studiolight_default` builds
//! (`source/blender/blenkernel/intern/studiolight.cc`), with no ambient term, no world-space
//! rotation, and the specular pass on. The shading is `get_world_lighting` in
//! `source/blender/draw/engines/workbench/shaders/workbench_world_light.bsl.hh`. Its roughness and
//! metallic arrive through the material buffer and the G-buffer's float pair, which
//! [`quantized_roughness`] and [`quantized_metallic`] reproduce.

use glam::DVec3;

/// One of a studio light's four lights, in view space.
struct StudioLight {
    /// Unit direction towards the light (`SolidLight.vec`).
    direction: [f64; 3],
    /// Diffuse colour (`SolidLight.col`).
    diffuse: [f64; 3],
    /// Specular colour (`SolidLight.spec`).
    specular: [f64; 3],
    /// Wrap factor (`SolidLight.smooth`).
    wrap: f64,
}

/// The four lights of `BKE_studiolight_default`.
const DEFAULT_LIGHTS: [StudioLight; 4] = [
    StudioLight {
        direction: [-0.352_546, 0.170_931, -0.920_051],
        diffuse: [0.033_103; 3],
        specular: [0.266_761; 3],
        wrap: 0.526_620,
    },
    StudioLight {
        direction: [-0.408_163, 0.346_939, 0.844_415],
        diffuse: [0.521_083, 0.538_226, 0.538_226],
        specular: [0.599_030; 3],
        wrap: 0.0,
    },
    StudioLight {
        direction: [0.521_739, 0.826_087, 0.212_999],
        diffuse: [0.038_403, 0.034_357, 0.049_530],
        specular: [0.106_102, 0.125_981, 0.158_523],
        wrap: 0.478_261,
    },
    StudioLight {
        direction: [0.624_519, -0.562_067, -0.542_269],
        diffuse: [0.090_838, 0.082_080, 0.072_255],
        specular: [0.106_535, 0.084_771, 0.066_080],
        wrap: 0.2,
    },
];

/// Workbench's `fast_rcp` (`workbench_world_light.bsl.hh`): a bit-trick reciprocal in single
/// precision. It is an approximation, and the lighting is defined with it, so it is kept.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Workbench evaluates its reciprocal in single precision"
)]
fn fast_reciprocal(value: f64) -> f64 {
    let bits = 0x7eef_370b_u32.wrapping_sub((value as f32).to_bits());
    f64::from(f32::from_bits(bits))
}

/// `unit_float_to_uchar_clamp` followed by a division by 255: the value an 8-bit unit channel holds.
fn unit_byte(value: f64) -> f64 {
    (255.0 * value.clamp(0.0, 1.0) + 0.5).floor() / 255.0
}

/// Roughness as the shader reads it. `Material::pack_data` stores `sqrt(roughness)` in 8 bits, and
/// `float_pair_encode` keeps 5 bits of it in the G-buffer.
fn quantized_roughness(roughness: f64) -> f64 {
    (unit_byte(roughness.clamp(0.0, 1.0).sqrt()) * 31.0).floor() / 31.0
}

/// Metallic as the shader reads it: 8 bits in the material buffer, then 3 bits in the G-buffer.
fn quantized_metallic(metallic: f64) -> f64 {
    (unit_byte(metallic) * 7.0).floor() / 7.0
}

/// `wrapped_lighting`: a Lambert term with the light's wrap factor, clamped to [0, 1].
fn wrapped_lighting(cosine: f64, wrap: f64) -> f64 {
    let wrap_plus_one = wrap + 1.0;
    ((cosine + wrap) * fast_reciprocal(wrap_plus_one * wrap_plus_one)).clamp(0.0, 1.0)
}

/// Workbench's studio lighting of a surface point, in linear RGB.
///
/// `albedo` is the base colour. `roughness` and `metallic` are the material's values, before
/// quantization. `normal` is the unit normal turned toward the viewer, and `to_eye` the unit
/// direction from the point to the viewer. Both are in view space, where the lights are fixed.
pub(crate) fn shade(
    albedo: DVec3,
    roughness: f64,
    metallic: f64,
    normal: DVec3,
    to_eye: DVec3,
) -> DVec3 {
    let roughness = quantized_roughness(roughness);
    let metallic = quantized_metallic(metallic);
    let diffuse_color = albedo * (1.0 - metallic);
    let specular_color = DVec3::splat(0.05) * (1.0 - metallic) + albedo * metallic;
    let normal_dot_eye = normal.dot(to_eye);
    // The view ray reflected about the normal, with the sign the shader uses (`-reflect(I, N)`).
    let reflected = normal * (2.0 * normal_dot_eye) - to_eye;

    let mut specular_light = DVec3::ZERO;
    let mut diffuse_light = DVec3::ZERO;
    for light in &DEFAULT_LIGHTS {
        let direction = DVec3::from_array(light.direction);
        let wrap = light.wrap;
        // A Blinn lobe, blended toward the wrapped environment term by the squared wrap factor.
        let half = (direction + to_eye).normalize_or_zero();
        let spec_angle = half.dot(normal).clamp(0.0, 1.0);
        let spec_nl = direction.dot(normal).clamp(0.0, 1.0);
        let gloss = (1.0 - roughness) * (1.0 - wrap);
        let shininess = (10.0 * gloss + 1.0).exp2();
        let blinn = spec_angle.powf(shininess) * spec_nl * (shininess * 0.125 + 1.0);
        let environment_wrap = wrap * (1.0 - roughness) + roughness;
        let environment = wrapped_lighting(direction.dot(reflected), environment_wrap);
        let spec_light = blinn * (1.0 - wrap * wrap) + environment * wrap * wrap;
        specular_light += DVec3::from_array(light.specular) * spec_light;

        let diffuse = wrapped_lighting(direction.dot(normal), wrap);
        diffuse_light += DVec3::from_array(light.diffuse) * diffuse;
    }

    // `brdf_approx` brightens the specular colour toward white at grazing view angles.
    let fresnel = (-8.35 * normal_dot_eye.clamp(0.0, 1.0)).exp2() * (1.0 - roughness);
    let specular_color = specular_color + (DVec3::ONE - specular_color) * fresnel;
    let specular_light = specular_light * specular_color;
    // Energy conservation: the diffuse term yields to the specular colour's mean.
    let spec_energy = specular_color.dot(DVec3::splat(0.33333));
    diffuse_light * diffuse_color * (1.0 - spec_energy) + specular_light
}

#[cfg(test)]
mod tests {
    use super::{quantized_metallic, quantized_roughness};

    #[test]
    fn material_values_keep_workbench_precision() {
        // sqrt(0.4) is 161/255 in 8 bits, which keeps 19 of the 31 steps of the 5-bit channel.
        assert!((quantized_roughness(0.4) - 19.0 / 31.0).abs() < 1.0e-12);
        // Metallic 0.5 is 128/255 in 8 bits, which keeps 3 of the 7 steps of the 3-bit channel.
        assert!((quantized_metallic(0.5) - 3.0 / 7.0).abs() < 1.0e-12);
        assert!(quantized_metallic(0.0).abs() < 1.0e-12);
    }
}
