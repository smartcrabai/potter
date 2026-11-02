//! CPU port of Blender 5.2.2 EEVEE `ray_generate_direction`.

use glam::{Mat3, Vec2, Vec3};

const PI: f32 = std::f32::consts::PI;
const INV_PI: f32 = std::f32::consts::FRAC_1_PI;
const BSDF_ROUGHNESS_THRESHOLD: f32 = 2.0e-2_f32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RayLobe {
    None,
    Diffuse,
    Translucent,
    BurleyBssrdf,
    GgxReflection,
    GgxRefraction,
    ThinGlassTransmission,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ThicknessMode {
    Slab,
    Sphere,
}

/// EEVEE thickness model and non-negative thickness value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RayThickness {
    pub(crate) mode: ThicknessMode,
    pub(crate) value: f32,
}

impl RayThickness {
    fn source_mode(self) -> ThicknessMode {
        if self.value.partial_cmp(&0.0) == Some(std::cmp::Ordering::Equal) {
            /* EEVEE's signed-float encoding reports zero thickness as Slab. */
            ThicknessMode::Slab
        } else {
            self.mode
        }
    }
}

/// Inputs read from the selected EEVEE `GBuffer` closure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RayGenerateInput {
    pub(crate) lobe: RayLobe,
    pub(crate) normal: Vec3,
    /// EEVEE's view vector `V`, pointing toward the viewer.
    pub(crate) view: Vec3,
    pub(crate) roughness: f32,
    pub(crate) ior: f32,
    pub(crate) thickness: RayThickness,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RaySample {
    pub(crate) direction: Vec3,
    pub(crate) pdf: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RayGenerateError {
    InvalidInput,
    NoSampleableClosure,
}

/// Samples the same view-space direction and PDF as EEVEE's `ray_generate_direction`.
pub(crate) fn generate_ray(
    input: RayGenerateInput,
    noise: [f32; 2],
) -> Result<RaySample, RayGenerateError> {
    if input.lobe == RayLobe::None {
        return Err(RayGenerateError::NoSampleableClosure);
    }
    if !noise[0].is_finite()
        || !noise[1].is_finite()
        || !(0.0..1.0).contains(&noise[0])
        || !(0.0..1.0).contains(&noise[1])
        || !input.normal.is_finite()
        || input.normal.length_squared() == 0.0
        || !input.view.is_finite()
        || input.view.length_squared() == 0.0
        || !input.roughness.is_finite()
        || !(0.0..=1.0).contains(&input.roughness)
        || !input.ior.is_finite()
        || input.ior <= 0.0
        || !input.thickness.value.is_finite()
        || input.thickness.value < 0.0
    {
        return Err(RayGenerateError::InvalidInput);
    }

    let thickness = input.thickness;
    let mut random = sample_cylinder(Vec2::new(noise[0], noise[1]));
    let rng_bias = 0.08_f32;
    if input.lobe != RayLobe::Translucent || thickness.source_mode() == ThicknessMode::Slab {
        random.x = 1.0 - random.x * (1.0 - rng_bias);
    }

    let mut normal = input.normal;
    let mut view = input.view;
    if input.lobe == RayLobe::GgxRefraction && thickness.value != 0.0 {
        amend_transmission_context(
            &mut normal,
            &mut view,
            input.roughness,
            input.ior,
            thickness,
        );
    }

    let tangent_to_world = from_up_axis(normal);
    let view_tangent = tangent_to_world.transpose() * view;
    let mut sample = match input.lobe {
        RayLobe::None => unreachable!("rejected above"),
        RayLobe::Diffuse | RayLobe::BurleyBssrdf => diffuse_sample(random),
        RayLobe::Translucent => translucent_sample(random, thickness),
        RayLobe::GgxReflection => {
            ggx_sample_reflection(random, view_tangent, square(input.roughness))
        }
        RayLobe::GgxRefraction => ggx_sample_refraction(
            random,
            view_tangent,
            square(input.roughness),
            input.ior,
            thickness,
        ),
        RayLobe::ThinGlassTransmission => {
            let mut sample = ggx_sample_reflection(random, view_tangent, square(input.roughness));
            sample.direction.z = -sample.direction.z;
            sample
        }
    };
    sample.direction = tangent_to_world * sample.direction;
    Ok(sample)
}

fn sample_cylinder(noise: Vec2) -> Vec3 {
    let phi = (noise.y - 0.5) * (2.0 * PI);
    let cos_phi = phi.cos();
    let sin_phi = safe_sqrt(1.0 - square(cos_phi)) * sign(phi);
    Vec3::new(noise.x, cos_phi, sin_phi)
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

fn diffuse_sample(random: Vec3) -> RaySample {
    let cos_theta = safe_sqrt(random.x);
    let sin_theta = sin_from_cos(cos_theta);
    RaySample {
        direction: Vec3::new(random.y * sin_theta, random.z * sin_theta, cos_theta),
        pdf: cos_theta * INV_PI,
    }
}

fn translucent_sample(random: Vec3, thickness: RayThickness) -> RaySample {
    if thickness.source_mode() == ThicknessMode::Sphere && thickness.value != 0.0 {
        let cos_theta = random.x * 2.0 - 1.0;
        let sin_theta = sin_from_cos(cos_theta);
        RaySample {
            direction: Vec3::new(random.y * sin_theta, random.z * sin_theta, -cos_theta),
            pdf: 0.25 * INV_PI,
        }
    } else {
        let mut sample = diffuse_sample(random);
        sample.direction = -sample.direction;
        sample
    }
}

fn ggx_sample_reflection(random: Vec3, view: Vec3, alpha: f32) -> RaySample {
    if alpha < square(BSDF_ROUGHNESS_THRESHOLD) {
        return RaySample {
            pdf: 1.0e6_f32,
            direction: reflect(-view, Vec3::Z),
        };
    }

    let view_alpha = Vec3::new(alpha * view.x, alpha * view.y, view.z);
    let (view_half, view_half_length) = normalize_and_get_length(view_alpha);
    let alpha_squared = square(alpha);
    let s_squared = square(1.0 + view.truncate().length());
    let k = (1.0 - alpha_squared) * s_squared / (s_squared + alpha_squared * square(view.z));

    let cos_theta = mix(-view_half.z * k, 1.0, random.x);
    let sin_theta = safe_sqrt(saturate(1.0 - square(cos_theta)));
    let light_half = Vec3::new(sin_theta * random.y, sin_theta * random.z, cos_theta);
    let half_half = view_half + light_half;
    let half = normalize(Vec3::new(
        alpha * half_half.x,
        alpha * half_half.y,
        half_half.z.max(0.0),
    ));

    let distribution = ggx_distribution(saturate(half.z), alpha_squared);
    let pdf = if view.z >= 0.0 {
        distribution / (2.0 * (k * view.z + view_half_length))
    } else {
        let view_alpha_length_squared = view_alpha.length_squared();
        let t = safe_sqrt(view_alpha_length_squared + square(view.z));
        distribution * (t - view.z) / (2.0 * view_alpha_length_squared)
    };

    RaySample {
        direction: reflect(-view, half),
        pdf,
    }
}

fn ggx_sample_refraction(
    random: Vec3,
    view: Vec3,
    alpha: f32,
    mut ior: f32,
    thickness: RayThickness,
) -> RaySample {
    if thickness.value != 0.0 {
        ior = 1.0 / ior;
    }

    if alpha < square(BSDF_ROUGHNESS_THRESHOLD) {
        return RaySample {
            pdf: 1.0e6_f32,
            direction: refract(-view, Vec3::Z, 1.0 / ior),
        };
    }

    let view_half = normalize(Vec3::new(alpha * view.x, alpha * view.y, view.z));
    let geometry_view = 2.0 * view_half.z / (1.0 + view_half.z);

    let cos_theta = mix(-view_half.z, 1.0, random.x);
    let sin_theta = safe_sqrt(saturate(1.0 - square(cos_theta)));
    let light_half = Vec3::new(sin_theta * random.y, sin_theta * random.z, cos_theta);
    let half_half = view_half + light_half;
    let half = normalize(Vec3::new(
        alpha * half_half.x,
        alpha * half_half.y,
        half_half.z.max(0.0),
    ));
    let direction = refract(-view, half, 1.0 / ior);

    let distribution = ggx_distribution(saturate(half.z), square(alpha));
    let view_half_dot = view.dot(half);
    let light_half_dot = direction.dot(half);
    let denominator = square(ior * light_half_dot + view_half_dot);
    let pdf = distribution * geometry_view * (view_half_dot * light_half_dot).abs() * square(ior)
        / (view.z * denominator);

    RaySample { direction, pdf }
}

fn ggx_distribution(normal_half: f32, alpha_squared: f32) -> f32 {
    let normal_half_squared = square(normal_half);
    alpha_squared / (PI * square((1.0 - normal_half_squared) + alpha_squared * normal_half_squared))
}

fn amend_transmission_context(
    normal: &mut Vec3,
    view: &mut Vec3,
    roughness: f32,
    ior: f32,
    thickness: RayThickness,
) {
    let perceived_roughness_factor = if ior > 1.0 {
        1.0 - 1.0 / ior
    } else {
        saturate(1.0 - ior) * 0.64
    };
    let perceived_roughness = roughness * sqrt_fast(perceived_roughness_factor);
    let direction = dominant_transmission_direction(*normal, *view, ior, perceived_roughness);
    if thickness.source_mode() == ThicknessMode::Sphere {
        let cos_alpha = direction.dot(-*normal);
        let hit_normal = normalize(*normal + direction * (cos_alpha * 2.0));
        *normal = -hit_normal;
    }
    *view = -direction;
}

fn dominant_transmission_direction(normal: Vec3, view: Vec3, ior: f32, roughness: f32) -> Vec3 {
    let m = square(roughness);
    let refracted = refract(-view, normal, 1.0 / ior);
    let smoothness = 1.0 - m;
    let factor = smoothness * (smoothness.sqrt() + m);
    normalize(mix_vec3(-normal, refracted, factor))
}

fn from_up_axis(up: Vec3) -> Mat3 {
    let z_sign = if up.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (z_sign + up.z);
    let b = up.x * up.y * a;
    let x_axis = Vec3::new(1.0 + z_sign * square(up.x) * a, z_sign * b, -z_sign * up.x);
    let y_axis = Vec3::new(b, z_sign + square(up.y) * a, -up.y);
    Mat3::from_cols(x_axis, y_axis, up)
}

fn normalize_and_get_length(vector: Vec3) -> (Vec3, f32) {
    let mut length = vector.length_squared();
    if length > 1.0e-35 {
        length = length.sqrt();
        (vector / length, length)
    } else {
        (Vec3::ZERO, 0.0)
    }
}

fn normalize(vector: Vec3) -> Vec3 {
    vector / vector.length()
}

fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    incident - 2.0 * normal.dot(incident) * normal
}

fn refract(incident: Vec3, normal: Vec3, eta: f32) -> Vec3 {
    let normal_incident = normal.dot(incident);
    let k = 1.0 - eta * eta * (1.0 - square(normal_incident));
    if k < 0.0 {
        Vec3::ZERO
    } else {
        eta * incident - (eta * normal_incident + k.sqrt()) * normal
    }
}

fn safe_sqrt(value: f32) -> f32 {
    value.max(0.0).sqrt()
}

fn sin_from_cos(cosine: f32) -> f32 {
    safe_sqrt(1.0 - square(cosine))
}

fn saturate(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn square(value: f32) -> f32 {
    value * value
}

fn mix(a: f32, b: f32, factor: f32) -> f32 {
    a * (1.0 - factor) + b * factor
}

fn mix_vec3(a: Vec3, b: Vec3, factor: f32) -> Vec3 {
    a * (1.0 - factor) + b * factor
}

fn sqrt_fast(value: f32) -> f32 {
    f32::from_bits(0x1fbd_1df5_u32.wrapping_add(value.to_bits() >> 1))
}
