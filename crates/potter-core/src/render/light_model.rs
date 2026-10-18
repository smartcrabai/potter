//! Light conventions of Cycles at Blender v5.2.2, shared by the path tracer and the realtime shading.
//!
//! - `intern/cycles/scene/light.cpp`: `PointLight::copy_to_kernel` sets `eval_fac = invarea / π`
//!   with `area = 4π r²` (and `4` when `r = 0`), so a point or spot light of radiant flux `P` has
//!   radiant intensity `P / (4π)`. `AreaLight::copy_to_kernel` normalises by the emitter area, so an
//!   area light of flux `P` has radiance `P / (π · area)`.
//! - `intern/cycles/kernel/light/spot.h`: `spot_light_attenuation` is
//!   `smoothstep((cos θ − cos(angle / 2)) · spot_smooth)` with
//!   `spot_smooth = 1 / ((1 − cos(angle / 2)) · blend)`.
//! - `intern/cycles/scene/light.cpp`, `AreaLight::area`: a rectangle's area is its extents, and an
//!   ellipse's (`DISK`, `ELLIPSE`) is `π/4` of its bounding rectangle.

use std::f64::consts::{PI, TAU};

use glam::{DMat4, DVec3};

use crate::model::LightData;

/// Segments of the inscribed polygon that stands in for an ellipse emitter.
const ELLIPSE_SEGMENTS: usize = 128;
/// Room for an outline plus the one vertex that a convex half-space clip adds.
const OUTLINE_CAPACITY: usize = ELLIPSE_SEGMENTS + 1;

/// Radiant intensity (W/sr) of a point or spot light with radiant flux `energy` (W).
pub(crate) fn point_intensity(energy: f64) -> f64 {
    energy / (4.0 * PI)
}

/// Cycles' `eval_fac` for a sphere or disk light of radius `radius` with radiant flux `energy`:
/// `invarea / π` with `invarea = 1 / (4π r²)`. Radiance from the emitter is this value.
pub(crate) fn emitter_radiance(energy: f64, radius: f64) -> f64 {
    energy / (4.0 * PI * PI * radius * radius)
}

/// Radiance (W/(sr·m²)) of an area light with radiant flux `energy` over an emitter of `area`.
pub(crate) fn area_radiance(energy: f64, area: f64) -> f64 {
    energy / (PI * area)
}

/// The spot cone's angular attenuation for a direction `cos_axis` = cos θ between the spot axis
/// and the direction from the light towards the receiver (`spot_light_attenuation`).
pub(crate) fn spot_attenuation(cos_axis: f64, spot_size: f64, spot_blend: f64) -> f64 {
    let cos_half = (spot_size.clamp(0.0, PI) * 0.5).cos();
    let blend = spot_blend.clamp(0.0, 1.0);
    let spread = (1.0 - cos_half) * blend;
    let x = if spread > 0.0 {
        (cos_axis - cos_half) / spread
    } else if cos_axis > cos_half {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The shape of a Blender area light in world space: unit axes, half extents with the object
/// scale applied, whether it is an ellipse, and its area.
///
/// Both sides take `area_size`, except that RECTANGLE and ELLIPSE take `area_size_y` for the
/// second. DISK and ELLIPSE are ellipses, whose area is `π/4` of the bounding rectangle
/// (`AreaLight::area`). The object transform scales the extents.
#[derive(Clone, Copy)]
pub(crate) struct AreaShape {
    pub(crate) axis_u: DVec3,
    pub(crate) axis_v: DVec3,
    pub(crate) half_u: f64,
    pub(crate) half_v: f64,
    pub(crate) ellipse: bool,
    pub(crate) area: f64,
}

/// Builds the world-space shape of an area light from its data and object transform
/// (`intern/cycles/blender/light.cpp`, `BlenderSync::sync_light`).
pub(crate) fn area_shape(data: &LightData, matrix: &DMat4) -> AreaShape {
    let square_or_disk = matches!(data.area_shape.as_str(), "SQUARE" | "DISK");
    let size_v = if square_or_disk {
        data.area_size
    } else {
        data.area_size_y
    };
    let extent_u = matrix.transform_vector3(DVec3::X) * data.area_size;
    let extent_v = matrix.transform_vector3(DVec3::Y) * size_v;
    let half_u = extent_u.length() * 0.5;
    let half_v = extent_v.length() * 0.5;
    let ellipse = matches!(data.area_shape.as_str(), "DISK" | "ELLIPSE");
    let rectangle_area = 4.0 * half_u * half_v;
    AreaShape {
        axis_u: extent_u.normalize_or_zero(),
        axis_v: extent_v.normalize_or_zero(),
        half_u,
        half_v,
        ellipse,
        area: if ellipse {
            rectangle_area * std::f64::consts::FRAC_PI_4
        } else {
            rectangle_area
        },
    }
}

/// Writes the emitter outline into `out`: the four corners of a rectangle, or the inscribed
/// polygon of an ellipse. The order makes the polygon normal oppose `light_direction`, the
/// emitting direction, which is what the irradiance form below requires. Returns the count.
fn emitter_outline(
    shape: &AreaShape,
    centre: DVec3,
    light_direction: DVec3,
    out: &mut [DVec3; OUTLINE_CAPACITY],
) -> usize {
    let corner = |su: f64, sv: f64| {
        centre + shape.axis_u * (shape.half_u * su) + shape.axis_v * (shape.half_v * sv)
    };
    let count = if shape.ellipse {
        for (index, slot) in out.iter_mut().take(ELLIPSE_SEGMENTS).enumerate() {
            let angle = TAU * index as f64 / ELLIPSE_SEGMENTS as f64;
            *slot = corner(angle.cos(), angle.sin());
        }
        ELLIPSE_SEGMENTS
    } else {
        out[0] = corner(-1.0, -1.0);
        out[1] = corner(1.0, -1.0);
        out[2] = corner(1.0, 1.0);
        out[3] = corner(-1.0, 1.0);
        4
    };
    let normal = (out[1] - out[0]).cross(out[2] - out[1]);
    if normal.dot(light_direction) > 0.0 {
        out[..count].reverse();
    }
    count
}

/// Clips a convex polygon to the half-space `normal · (x − point) ≥ 0` (Sutherland–Hodgman).
fn clip_to_receiver(
    input: &[DVec3],
    point: DVec3,
    normal: DVec3,
    output: &mut [DVec3; OUTLINE_CAPACITY],
) -> usize {
    let mut count = 0;
    let mut push = |vertex: DVec3| {
        if let Some(slot) = output.get_mut(count) {
            *slot = vertex;
            count += 1;
        }
    };
    let Some(&last) = input.last() else {
        return 0;
    };
    let mut previous = last;
    let mut previous_side = normal.dot(previous - point);
    for &current in input {
        let side = normal.dot(current - point);
        let crosses = (side >= 0.0) != (previous_side >= 0.0);
        if crosses {
            let t = previous_side / (previous_side - side);
            push(previous + (current - previous) * t);
        }
        if side >= 0.0 {
            push(current);
        }
        previous = current;
        previous_side = side;
    }
    count
}

#[derive(Clone, Copy, Default)]
pub(crate) struct AreaVisibilitySample {
    pub(crate) direction: DVec3,
    pub(crate) distance: f64,
    /// Receiver cosine × emitter cosine / distance²; zero-contribution directions are excluded.
    pub(crate) weight: f64,
}

/// Deterministic points for area visibility and diffuse Fresnel weighting. The centre and four
/// cross points are retained when they contribute; if all miss a positive horizon-clipped region,
/// its centroid supplies one contributing direction.
pub(crate) fn area_visibility_samples(
    shape: &AreaShape,
    centre: DVec3,
    light_direction: DVec3,
    point: DVec3,
    normal: DVec3,
) -> ([AreaVisibilitySample; 6], usize) {
    let mut outline = [DVec3::ZERO; OUTLINE_CAPACITY];
    let outline_count = emitter_outline(shape, centre, light_direction, &mut outline);
    let mut clipped = [DVec3::ZERO; OUTLINE_CAPACITY];
    let clipped_count = clip_to_receiver(&outline[..outline_count], point, normal, &mut clipped);
    if clipped_count < 3 {
        return ([AreaVisibilitySample::default(); 6], 0);
    }

    let cross_scale = if shape.ellipse { 0.5 } else { 1.0 };
    let offsets = [
        (0.0, 0.0),
        (cross_scale, 0.0),
        (-cross_scale, 0.0),
        (0.0, cross_scale),
        (0.0, -cross_scale),
    ];
    let mut samples = [AreaVisibilitySample::default(); 6];
    let mut count = 0;
    for (u, v) in offsets {
        let sample_point =
            centre + shape.axis_u * (shape.half_u * u) + shape.axis_v * (shape.half_v * v);
        let delta = sample_point - point;
        let distance_squared = delta.length_squared();
        if distance_squared <= f64::EPSILON {
            continue;
        }
        let distance = distance_squared.sqrt();
        let direction = delta / distance;
        let weight = normal.dot(direction).max(0.0) * light_direction.dot(-direction).max(0.0)
            / distance_squared;
        if weight > 0.0 {
            samples[count] = AreaVisibilitySample {
                direction,
                distance,
                weight,
            };
            count += 1;
        }
    }
    if count == 0 {
        let centroid = clipped[..clipped_count]
            .iter()
            .copied()
            .fold(DVec3::ZERO, |sum, vertex| sum + vertex)
            / clipped_count as f64;
        let delta = centroid - point;
        let distance_squared = delta.length_squared();
        if distance_squared > f64::EPSILON {
            let distance = distance_squared.sqrt();
            let direction = delta / distance;
            let weight = normal.dot(direction).max(0.0) * light_direction.dot(-direction).max(0.0)
                / distance_squared;
            if weight > 0.0 {
                samples[0] = AreaVisibilitySample {
                    direction,
                    distance,
                    weight,
                };
                count = 1;
            }
        }
    }
    (samples, count)
}

/// Lambert irradiance (W/m²) at `point` with receiver normal `normal`, per unit radiance of the
/// area light `shape` centred at `centre` and emitting along `light_direction`. The receiver's
/// cosine is included; the emitter is one-sided.
///
/// Rectangles are exact. The polygon form (Arvo, "Applications of Irradiance Tensors", 1995) gives
/// `E = ½ Σ θᵢ (n · γᵢ)` for an outline seen from `point`, where `θᵢ` is the angle subtended by
/// edge `i` and `γᵢ` the unit normal of the plane through `point` and that edge. The outline is
/// first clipped to the receiver's horizon, because the form assumes the emitter lies above it.
/// Ellipses use an inscribed polygon of 128 segments, which is 0.03% below the exact value on
/// axis; the radiance `P / (π · area)` still uses the ellipse's exact area.
pub(crate) fn area_irradiance_per_radiance(
    shape: &AreaShape,
    centre: DVec3,
    light_direction: DVec3,
    point: DVec3,
    normal: DVec3,
) -> f64 {
    let mut outline = [DVec3::ZERO; OUTLINE_CAPACITY];
    let count = emitter_outline(shape, centre, light_direction, &mut outline);
    let mut clipped = [DVec3::ZERO; OUTLINE_CAPACITY];
    let clipped_count = clip_to_receiver(&outline[..count], point, normal, &mut clipped);
    if clipped_count < 3 {
        return 0.0;
    }
    let vertices = &clipped[..clipped_count];
    let mut sum = 0.0;
    for (index, &vertex) in vertices.iter().enumerate() {
        let next = vertices[(index + 1) % clipped_count];
        let a = vertex - point;
        let b = next - point;
        let cross = a.cross(b);
        let length = cross.length();
        if length <= f64::EPSILON {
            continue;
        }
        let subtended = length.atan2(a.dot(b));
        sum += subtended * normal.dot(cross) / length;
    }
    (0.5 * sum).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disk_of_radius_one_at_height_two() -> AreaShape {
        AreaShape {
            axis_u: DVec3::X,
            axis_v: DVec3::Y,
            half_u: 1.0,
            half_v: 1.0,
            ellipse: true,
            area: PI,
        }
    }

    #[test]
    fn point_intensity_divides_flux_by_four_pi() {
        assert!((point_intensity(4.0 * PI) - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn spot_attenuation_has_a_hard_edge_at_zero_blend_and_a_full_smoothstep_at_unit_blend() {
        let half: f64 = 0.5;
        // blend 0: a hard edge at the half angle.
        assert_eq!(spot_attenuation(half.cos() + 1.0e-6, 2.0 * half, 0.0), 1.0);
        assert_eq!(spot_attenuation(half.cos() - 1.0e-6, 2.0 * half, 0.0), 0.0);
        // blend 1: a smoothstep over the whole cone, zero at the axis edge.
        assert_eq!(spot_attenuation(1.0, 2.0 * half, 1.0), 1.0);
        assert_eq!(spot_attenuation(half.cos(), 2.0 * half, 1.0), 0.0);
    }

    #[test]
    fn spot_attenuation_matches_the_smoothstep_of_cycles() {
        let (size, blend) = (1.2_f64, 0.4_f64);
        let cos_half = (size * 0.5).cos();
        let cos_axis = cos_half + 0.5 * (1.0 - cos_half) * blend;
        // Halfway through the blend band the smoothstep is 0.5.
        let got = spot_attenuation(cos_axis, size, blend);
        assert!((got - 0.5).abs() < 1.0e-12, "got {got}");
    }

    #[test]
    fn disk_irradiance_on_axis_matches_the_closed_form() {
        // A uniform disk of radius R and radiance 1 at height h gives E = π R² / (R² + h²).
        let shape = disk_of_radius_one_at_height_two();
        let centre = DVec3::new(0.0, 0.0, 2.0);
        let irradiance =
            area_irradiance_per_radiance(&shape, centre, DVec3::NEG_Z, DVec3::ZERO, DVec3::Z);
        let exact = PI / 5.0;
        assert!(
            (irradiance - exact).abs() <= 0.002 * exact,
            "irradiance {irradiance} against exact {exact}"
        );
    }

    #[test]
    fn square_irradiance_matches_numerical_integration() {
        // A 2 m square (half extent 1) at height 2 under the receiver: numerically integrated
        // as Σ L cos θ_receiver cos θ_light / d² dA, which gives 0.752276.
        let shape = AreaShape {
            axis_u: DVec3::X,
            axis_v: DVec3::Y,
            half_u: 1.0,
            half_v: 1.0,
            ellipse: false,
            area: 4.0,
        };
        let irradiance = area_irradiance_per_radiance(
            &shape,
            DVec3::new(0.0, 0.0, 2.0),
            DVec3::NEG_Z,
            DVec3::ZERO,
            DVec3::Z,
        );
        assert!(
            (irradiance - 0.752_276).abs() < 1.0e-4,
            "irradiance {irradiance}"
        );
    }

    fn tilted_horizon_crossing_shape(ellipse: bool) -> (AreaShape, DVec3, DVec3) {
        let axis_v = DVec3::new(
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        );
        let light_direction = DVec3::X.cross(axis_v);
        (
            AreaShape {
                axis_u: DVec3::X,
                axis_v,
                half_u: 1.0,
                half_v: 2.0,
                ellipse,
                area: if ellipse { 2.0 * PI } else { 8.0 },
            },
            DVec3::new(0.0, 0.0, -1.0),
            light_direction,
        )
    }

    #[test]
    fn horizon_crossing_area_keeps_irradiance_and_excludes_below_horizon_samples() {
        let (shape, centre, light_direction) = tilted_horizon_crossing_shape(false);
        let irradiance =
            area_irradiance_per_radiance(&shape, centre, light_direction, DVec3::ZERO, DVec3::Z);
        assert!(
            irradiance > 0.0,
            "the upper strip must illuminate the receiver"
        );
        let (samples, count) =
            area_visibility_samples(&shape, centre, light_direction, DVec3::ZERO, DVec3::Z);
        assert!(count > 0);
        assert!(
            samples[..count]
                .iter()
                .all(|sample| sample.weight > 0.0 && sample.direction.z > 0.0)
        );
    }

    #[test]
    fn clipped_area_centroid_samples_a_visible_ellipse_strip_when_cross_points_miss() {
        let (shape, centre, light_direction) = tilted_horizon_crossing_shape(true);
        let irradiance =
            area_irradiance_per_radiance(&shape, centre, light_direction, DVec3::ZERO, DVec3::Z);
        assert!(
            irradiance > 0.0,
            "the upper ellipse strip must illuminate the receiver"
        );
        let (samples, count) =
            area_visibility_samples(&shape, centre, light_direction, DVec3::ZERO, DVec3::Z);
        assert_eq!(
            count, 1,
            "all five fixed ellipse samples miss; use the clipped centroid"
        );
        assert!(samples[0].weight > 0.0 && samples[0].direction.z > 0.0);
    }

    #[test]
    fn a_receiver_facing_away_from_the_emitter_gets_nothing() {
        let shape = disk_of_radius_one_at_height_two();
        let irradiance = area_irradiance_per_radiance(
            &shape,
            DVec3::new(0.0, 0.0, 2.0),
            DVec3::NEG_Z,
            DVec3::new(0.0, 0.0, 4.0),
            DVec3::NEG_Z,
        );
        assert!(irradiance.abs() < 1.0e-12, "irradiance {irradiance}");
    }
}
