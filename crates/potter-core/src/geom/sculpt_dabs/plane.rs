//! Blender 5.2.2 PLANE brushes: Flatten/Contrast, Fill/Deepen, and Scrape/Fill (`brushes/plane.cc`).
//!
//! Each dab fits a plane to the vertices around it (`calc_brush_plane`, AREA) and moves every vertex
//! along the plane normal by its signed height above the plane, times the falloff of its distance
//! from the plane center. The three brushes run this code and differ only in the asset values of
//! `FLATTEN`, `FILL`, and `SCRAPE`.
//!
//! Potter's stroke has no plane settings, so these Blender 5.2.2 essentials values are fixed here:
//! - `plane_offset` 0 and `tilt_strength_factor` 0: the plane center is not shifted and the normal
//!   is not tilted (`do_plane_brush`, `plane.cc:385-390`).
//! - `use_plane_trim` off: the trim filter (`filter_plane_trim_limit_factors`, `sculpt.cc:8298`) is
//!   gated by `BRUSH_PLANE_TRIM`, and `plane.cc` never calls it.
//! - `use_original_plane` and `use_original_normal` off, `use_accumulate` on: `calc_brush_plane`
//!   recomputes the plane on every main-pass dab from the current positions (`sculpt.cc:3060-3066`).
//! - `stabilize_normal` and `stabilize_plane` 0: `calc_stabilized_plane` (`sculpt.cc:2101`) returns
//!   the raw plane, so no stroke-start plane history is kept.
//! - `plane_inversion_mode`: read only when the stroke is flipped (`plane.cc:408`). Potter never
//!   flips a stroke, so the mode has no effect.
//! - No texture, automasking, or front-face test. The falloff is `field.falloff`; Blender's curve
//!   preset is SMOOTH for all three brushes.

use glam::DVec3;

use super::{DabInput, Field, area_plane};
use crate::geom::sculpt::DabBrush;

/// The asset values one plane brush reads.
#[derive(Clone, Copy)]
struct PlaneSettings {
    /// `normal_radius_factor`: the area normal radius is `radius * normal_radius_factor`
    /// (`area_normal_and_center_get_normal_radius`, `sculpt.cc:1438`).
    normal_radius_factor: f64,
    /// `area_radius_factor`: the area center radius is `radius * area_radius_factor`
    /// (`area_normal_and_center_get_position_radius`, `sculpt.cc:1447`).
    area_radius_factor: f64,
    /// `use_pressure_area_radius`: the area center radius is also scaled by the dab's pressure.
    area_radius_pressure: bool,
    /// `plane_height`: scales the distance of vertices above the plane (`calc_local_distances`).
    height: f64,
    /// `plane_depth`: scales the distance of vertices below the plane.
    depth: f64,
}

/// Flatten/Contrast: `plane_height` 1 and `plane_depth` 1, so vertices on both sides move toward the
/// plane. Its `plane_inversion_mode` is `INVERT_DISPLACEMENT`, which needs a flipped stroke.
const FLATTEN: PlaneSettings = PlaneSettings {
    normal_radius_factor: 0.75,
    area_radius_factor: 1.0,
    area_radius_pressure: false,
    height: 1.0,
    depth: 1.0,
};

/// Fill/Deepen: `plane_height` 0 and `plane_depth` 1, so only vertices below the plane move up to it.
const FILL: PlaneSettings = PlaneSettings {
    normal_radius_factor: 1.0,
    area_radius_factor: 1.0,
    area_radius_pressure: false,
    height: 0.0,
    depth: 1.0,
};

/// Scrape/Fill: `plane_height` 1 and `plane_depth` 0, so only vertices above the plane move down to
/// it. Its area center radius follows the pressure. Fill and Scrape share one plane code path and
/// differ in these values only.
const SCRAPE: PlaneSettings = PlaneSettings {
    normal_radius_factor: 0.5,
    area_radius_factor: 0.5,
    area_radius_pressure: true,
    height: 1.0,
    depth: 0.0,
};

/// Blender 5.2.2 Flatten, Fill, and Scrape for one dab. Every symmetry pass runs inside
/// [`plane_dab`].
pub(super) fn apply(field: &mut Field, dab: DabBrush, input: &DabInput) {
    let settings = match dab {
        DabBrush::Flatten => FLATTEN,
        DabBrush::Fill => FILL,
        DabBrush::Scrape => SCRAPE,
        _ => unreachable!("plane::apply receives only Flatten, Fill, and Scrape"),
    };
    plane_dab(field, settings, input);
}

/// Blender's `do_plane_brush` (`plane.cc:366`) for one dab. The area plane is computed once from
/// the main pass, before any pass moves a vertex. Each mirror pass flips that normal and center
/// rather than recomputing them (`calc_brush_plane`, `sculpt.cc:3115-3124`).
fn plane_dab(field: &mut Field, settings: PlaneSettings, dab: &DabInput) {
    // plane.cc:378: a zero grab delta moves nothing. A stroke's first dab has no derived grab.
    if dab.grab == DVec3::ZERO {
        return;
    }
    let Some(area) = area_plane(
        field,
        dab.center,
        dab.radius * settings.normal_radius_factor,
        position_radius(settings, dab),
        true,
    ) else {
        return;
    };
    for sign in &field.passes {
        let normal = area.normal * *sign;
        let center = area.center * *sign;
        let grab = dab.grab * *sign;
        // plane.cc:393-397: the frame's x axis is normal x grab. A grab parallel to the normal gives
        // a singular frame, which `math::invert` turns into a zero matrix, so no vertex moves.
        let x_axis = normal.cross(grab).normalize_or_zero();
        if x_axis == DVec3::ZERO {
            continue;
        }
        let y_axis = normal.cross(x_axis);
        for (index, position) in field.positions.iter_mut().enumerate() {
            let offset = *position - center;
            // `calc_local_positions`: the local z is the signed height in radii. The in-plane
            // length is the frame's x and y, which only enter through their length.
            let local_z = offset.dot(normal) / dab.radius;
            let in_plane = offset.dot(x_axis).hypot(offset.dot(y_axis)) / dab.radius;
            // `scale_factors_by_height_and_depth`: the side of the plane picks the scale.
            let side = if local_z >= 0.0 {
                settings.height
            } else {
                settings.depth
            };
            let distance = plane_distance(in_plane, local_z, side);
            // `filter_distances_with_radius(1.0)`: vertices at the falloff edge do not move.
            if distance >= 1.0 {
                continue;
            }
            // `calc_translations` (`plane.cc:178`): the factor carries the signed local z, and the
            // translation is `-normal * radius * strength * factor`, scaled by the mask.
            let mask = field.mask_factors[index];
            let factor = mask * field.falloff.weight(distance) * side * local_z;
            *position += normal * (-dab.strength * dab.radius * factor);
        }
    }
}

/// Blender's `area_normal_and_center_get_position_radius` (`sculpt.cc:1447`) for a PLANE brush: the
/// area radius factor, then the dab's pressure when `use_pressure_area_radius` is on.
fn position_radius(settings: PlaneSettings, dab: &DabInput) -> f64 {
    let radius = dab.radius * settings.area_radius_factor;
    if settings.area_radius_pressure {
        radius * dab.pressure
    } else {
        radius
    }
}

/// Blender's `calc_local_distances` (`plane.cc:80`) for one vertex: the in-plane length and the
/// signed height, with the height scaled by the side's height or depth. A zero scale leaves the
/// vertex at distance 1, outside the falloff.
fn plane_distance(in_plane: f64, local_z: f64, side: f64) -> f64 {
    if side == 0.0 {
        1.0
    } else {
        in_plane.hypot(local_z / side)
    }
}
