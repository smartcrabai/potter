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
//! - `intern/cycles/kernel/light/area.h`: `area_light_spread_attenuation` confines an area light's
//!   emission to a cone about its normal (`spread`), and `area_light_spread_clamp_light` makes light
//!   sampling draw from the part of the emitter that the cone reaches.

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

impl AreaShape {
    /// A shape with unit axes and half extents, whose area is `AreaLight::area`'s.
    pub(crate) fn new(
        axis_u: DVec3,
        axis_v: DVec3,
        half_u: f64,
        half_v: f64,
        ellipse: bool,
    ) -> Self {
        let rectangle_area = 4.0 * half_u * half_v;
        Self {
            axis_u,
            axis_v,
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
    AreaShape::new(
        extent_u.normalize_or_zero(),
        extent_v.normalize_or_zero(),
        extent_u.length() * 0.5,
        extent_v.length() * 0.5,
        matches!(data.area_shape.as_str(), "DISK" | "ELLIPSE"),
    )
}

/// A Blender area light's `spread` (Cycles' `AreaLight::spread`): the emission is confined to the
/// cone of half-angle `spread / 2` about the emitting normal. The trigonometry is computed once.
#[derive(Clone, Copy)]
pub(crate) struct Spread {
    /// Half the spread, in radians.
    half: f64,
    /// `tan(spread / 2)`.
    tan_half: f64,
    /// Cycles' normalisation in `spread_attenuation`: `1 / (tan a − a)` for the half-angle `a`, with
    /// its third-order form at small angles. Unused for a zero spread.
    normalize: f64,
}

impl Spread {
    /// The spread of a light with `spread` radians, or `None` at the default π, which leaves the
    /// emission and the sampling unchanged.
    pub(crate) fn new(spread: f64) -> Option<Self> {
        (spread < PI).then(|| {
            let half = 0.5 * spread.max(0.0);
            let tan_half = half.tan();
            let normalize = if half > 0.05 {
                1.0 / (tan_half - half)
            } else {
                3.0 / half.powi(3)
            };
            Self {
                half,
                tan_half,
                normalize,
            }
        })
    }

    /// Whether the cone is a line (`spread` 0), which Cycles samples as a delta.
    pub(crate) fn is_zero(self) -> bool {
        self.half == 0.0
    }
}

/// Cycles' `area_light_spread_attenuation`: the factor on the emitter radiance towards `to_receiver`,
/// the unit direction from the emitter point to the receiver, for an emitter with normal `normal`.
pub(crate) fn spread_attenuation(spread: Spread, to_receiver: DVec3, normal: DVec3) -> f64 {
    let cosine = normal.dot(to_receiver);
    if cosine <= 0.0 {
        return 0.0;
    }
    let tan_angle = normal.cross(to_receiver).length() / cosine;
    if spread.is_zero() {
        // Emission along the normal alone; π is the hemisphere integral that Cycles folds in.
        return if tan_angle > 1.0e-5 { 0.0 } else { PI };
    }
    ((spread.tan_half - tan_angle) * spread.normalize).max(0.0)
}

/// The part of an area light that light sampling draws from for `receiver`, as Cycles'
/// `area_light_spread_clamp_light`: the smallest of the emitter, a rectangle, and a disk that covers
/// where the spread cone meets the emitter. Points of a rectangle or disk outside the emitter emit
/// nothing, so sampling them only wastes samples. Returns the region's centre and shape, or `None`
/// when the receiver is not in front of the emitter or no part of it reaches the receiver.
pub(crate) fn spread_region(
    shape: &AreaShape,
    centre: DVec3,
    normal: DVec3,
    spread: Spread,
    receiver: DVec3,
) -> Option<(DVec3, AreaShape)> {
    let height = normal.dot(receiver - centre);
    if height <= 0.0 {
        return None;
    }
    // The foot of the receiver on the emitter's plane, and the radius of the disk on that plane
    // that the cone reaches.
    let closest = receiver - height * normal;
    let r_spread = height * spread.tan_half;
    let offset = closest - centre;
    let spread_u = shape.axis_u.dot(offset);
    let spread_v = shape.axis_v.dot(offset);
    let mut sample_spread = r_spread == 0.0;
    let is_round = shape.ellipse && shape.half_u - shape.half_v == 0.0;
    if is_round && !sample_spread {
        let r = shape.half_u;
        let dist = spread_u.hypot(spread_v);
        if dist >= r + r_spread {
            return None;
        }
        sample_spread = dist <= (r - r_spread).abs() && r_spread < r;
        if dist > (r - r_spread).abs() {
            // The two disks cross: the smallest rectangle covering their overlap, along the line of
            // centres.
            let len_u = r + r_spread - dist;
            let len_v = if (r * r - r_spread * r_spread).abs() >= dist * dist {
                2.0 * r.min(r_spread)
            } else {
                ((2.0 * r_spread).powi(2) - (dist + (r_spread * r_spread - r * r) / dist).powi(2))
                    .sqrt()
            };
            let circle_area = PI * r * r;
            let spread_area = PI * r_spread * r_spread;
            if len_u * len_v < circle_area.min(spread_area) {
                let axis_u = (centre - closest).normalize_or_zero();
                let axis_v = normal.cross(axis_u);
                let rectangle_centre = 0.5 * (centre + closest + axis_u * (r_spread - r));
                return Some((
                    rectangle_centre,
                    AreaShape::new(axis_u, axis_v, 0.5 * len_u, 0.5 * len_v, false),
                ));
            }
            sample_spread = spread_area < circle_area;
        }
    } else if !is_round && !sample_spread {
        // The rectangle that covers the disk, clipped to the emitter's own extents.
        let min_u = (spread_u - r_spread).max(-shape.half_u);
        let max_u = (spread_u + r_spread).min(shape.half_u);
        let min_v = (spread_v - r_spread).max(-shape.half_v);
        let max_v = (spread_v + r_spread).min(shape.half_v);
        if min_u >= max_u || min_v >= max_v {
            return None;
        }
        let rectangle_area = (max_u - min_u) * (max_v - min_v);
        let spread_area = PI * r_spread * r_spread;
        if !shape.ellipse || rectangle_area < shape.area.min(spread_area) {
            let rectangle_centre = centre
                + shape.axis_u * f64::midpoint(min_u, max_u)
                + shape.axis_v * f64::midpoint(min_v, max_v);
            return Some((
                rectangle_centre,
                AreaShape::new(
                    shape.axis_u,
                    shape.axis_v,
                    0.5 * (max_u - min_u),
                    0.5 * (max_v - min_v),
                    false,
                ),
            ));
        }
        sample_spread = spread_area < shape.area;
    }
    if sample_spread {
        let disk_centre = centre + shape.axis_u * spread_u + shape.axis_v * spread_v;
        return Some((
            disk_centre,
            AreaShape::new(shape.axis_u, shape.axis_v, r_spread, r_spread, true),
        ));
    }
    Some((centre, *shape))
}

/// Whether `point` lies on the emitter of `shape` centred at `centre`: inside its rectangle or ellipse.
pub(crate) fn inside_emitter(shape: &AreaShape, centre: DVec3, point: DVec3) -> bool {
    let offset = point - centre;
    let u = offset.dot(shape.axis_u) / shape.half_u;
    let v = offset.dot(shape.axis_v) / shape.half_v;
    if shape.ellipse {
        u * u + v * v <= 1.0
    } else {
        u.abs() <= 1.0 && v.abs() <= 1.0
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
