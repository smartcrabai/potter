//! Blender 5.2.2 Crease and Pinch for the dab framework. Both take their plane from the dab's area
//! normal (`calc_area_normal`, sculpt_plane AREA) at the asset's `normal_radius_factor`, and both
//! read the stroke-start positions because their assets do not accumulate.

use glam::DVec3;

use super::{DabInput, Field, area_plane};
use crate::geom::sculpt::DabBrush;

/// Crease Polish `normal_radius_factor` (asset value 0.5). A non-plane brush's normal radius is the
/// dab radius times this factor (`area_normal_and_center_get_normal_radius`, `sculpt.cc:1438`).
const CREASE_NORMAL_RADIUS_FACTOR: f64 = 0.5;
/// Crease Polish `crease_pinch_factor` (asset value 0.5), `brush.crease_pinch_factor`.
const CREASE_PINCH_FACTOR: f64 = 0.5;
/// Crease Polish `use_accumulate` (off): the area normal reads the stroke-start positions.
const CREASE_ACCUMULATE: bool = false;
/// Pinch/Magnify `normal_radius_factor` (asset value 0.5).
const PINCH_NORMAL_RADIUS_FACTOR: f64 = 0.5;
/// Pinch/Magnify `use_accumulate` (off): the area normal reads the stroke-start positions.
const PINCH_ACCUMULATE: bool = false;

/// Applies one Crease or Pinch dab; other variants never reach this module.
pub(super) fn apply(field: &mut Field, dab: DabBrush, input: &DabInput) {
    match dab {
        DabBrush::Crease => crease_dab(field, input),
        DabBrush::Pinch => pinch_dab(field, input),
        _ => unreachable!("crease_pinch::apply only receives Crease and Pinch"),
    }
}

/// Blender's `calc_sculpt_normal` for the AREA plane (`sculpt.cc:2706`): the normal of
/// `calc_area_normal`, or the zero vector when that normal sum is zero (`value_or(float3(0))`).
/// Only the normal is read; the position radius of the area center is not used by these brushes.
fn area_normal(field: &Field, center: DVec3, normal_radius: f64, accumulate: bool) -> DVec3 {
    area_plane(field, center, normal_radius, normal_radius, accumulate)
        .map_or(DVec3::ZERO, |area| area.normal)
}

/// Blender's `project_translations` (`sculpt.cc:7705`): removes the component of `translation`
/// along `plane`. A plane shorter than Blender's float epsilon leaves the translation unchanged.
fn project_out(translation: DVec3, plane: DVec3) -> DVec3 {
    let epsilon = f64::from(f32::EPSILON);
    let length_sq = plane.length_squared();
    if length_sq < epsilon {
        return translation;
    }
    translation - plane * (translation.dot(plane) / length_sq)
}

/// Blender's crease strength (`crease.cc:209-217`): `|bstrength|` times `crease_correction`, where
/// the correction is `crease_pinch_factor^2 / brush_alpha^2` for a positive brush alpha and
/// `crease_pinch_factor^2` otherwise. `bstrength` carries `brush_alpha^2`, so the product is
/// `pressure * feather * crease_pinch_factor^2`.
fn crease_strength(dab: &DabInput) -> f64 {
    let pinch_squared = CREASE_PINCH_FACTOR * CREASE_PINCH_FACTOR;
    let correction = if dab.brush_alpha > 0.0 {
        pinch_squared / (dab.brush_alpha * dab.brush_alpha)
    } else {
        pinch_squared
    };
    dab.strength.abs() * correction
}

/// Blender 5.2.2 Crease for one dab (`crease.cc`, `do_crease_or_blob_brush` with
/// `invert_strength` false, reached from `do_crease_brush`). Each vertex inside the dab moves by:
/// - `factor * strength * project(location - position)`, the pull toward the dab center with the
///   component along the normal removed (`calc_faces`, `crease.cc:99-112`);
/// - `factor * offset`, where `offset = normal * radius * bstrength` (`crease.cc:205`).
///
/// The normal is the area normal of the main pass, computed once before any pass moves vertices
/// (`update_sculpt_normal`, `sculpt.cc:2720-2757`). A mirror pass uses that normal flipped
/// (`symmetry_flip`, `sculpt.cc:2754`). Object scale is identity in Potter, so Blender's
/// `cache.scale` is 1.
fn crease_dab(field: &mut Field, dab: &DabInput) {
    let normal = area_normal(
        field,
        dab.center,
        dab.radius * CREASE_NORMAL_RADIUS_FACTOR,
        CREASE_ACCUMULATE,
    );
    let strength = crease_strength(dab);
    for sign in &field.passes {
        let location = dab.center * *sign;
        let plane = normal * *sign;
        let offset = plane * (dab.radius * dab.strength);
        for (index, position) in field.positions.iter_mut().enumerate() {
            let distance = position.distance(location);
            if distance >= dab.radius {
                continue;
            }
            let factor = field.falloff.weight(distance / dab.radius) * field.mask_factors[index];
            if factor == 0.0 {
                continue;
            }
            let pull = project_out((location - *position) * (factor * strength), plane);
            *position += pull + offset * factor;
        }
    }
}

/// Blender 5.2.2 Pinch for one dab (`pinch.cc`, `do_pinch_brush`). The frame is x = normal × grab,
/// y = normal × x, z = normal (`pinch.cc:197-206`). Each vertex inside the dab moves by
/// `factor * bstrength * (x (d·x) + z (d·z))`, with `d = location - position`; the Y component is
/// dropped (`calc_translations`, `pinch.cc:39-61`). `factor` already includes the falloff and mask.
///
/// The first dab has no grab direction, and a zero grab returns before any vertex moves
/// (`pinch.cc:189-195`); the zero-grab check here covers both. Mirror passes flip the normal and
/// the grab (`calc_brush_plane`, `sculpt.cc:3115-3124`).
fn pinch_dab(field: &mut Field, dab: &DabInput) {
    if dab.grab == DVec3::ZERO {
        return;
    }
    let normal = area_normal(
        field,
        dab.center,
        dab.radius * PINCH_NORMAL_RADIUS_FACTOR,
        PINCH_ACCUMULATE,
    );
    for sign in &field.passes {
        let plane = normal * *sign;
        let location = dab.center * *sign;
        let x_axis = plane.cross(dab.grab * *sign).normalize_or_zero();
        for (index, position) in field.positions.iter_mut().enumerate() {
            let distance = position.distance(location);
            if distance >= dab.radius {
                continue;
            }
            let factor = field.falloff.weight(distance / dab.radius) * field.mask_factors[index];
            if factor == 0.0 {
                continue;
            }
            let disp = location - *position;
            let translation = x_axis * disp.dot(x_axis) + plane * disp.dot(plane);
            *position += translation * (factor * dab.strength);
        }
    }
}
