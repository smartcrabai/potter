//! Blender 5.2.2 offset brushes: Draw, Draw Sharp, Blob, and Nudge. Each dab moves every vertex inside
//! the brush sphere by an offset scaled by the falloff and mask factors (`mesh_brush_common.cc`
//! `calc_factors_common_mesh_indexed` and `translations_from_offset_and_factors`). Draw, Draw Sharp,
//! and Blob offset along the area normal; Nudge slides along the grab direction within the surface.

use glam::DVec3;

use super::{DabInput, Field, area_plane};
use crate::geom::sculpt::DabBrush;

/// `normal_radius_factor` of the Draw, Draw Sharp, and Blob essentials assets. Blender sets the area
/// normal radius and the area position radius to `cache.radius * normal_radius_factor` for these
/// brushes (`area_normal_and_center_get_normal_radius`, `sculpt.cc` 1438-1460).
const DRAW_NORMAL_RADIUS_FACTOR: f64 = 0.5;
/// `normal_radius_factor` of the Nudge essentials asset (0.3; the asset stores it as f32).
const NUDGE_NORMAL_RADIUS_FACTOR: f64 = 0.3;
/// `crease_pinch_factor` of the Blob essentials asset (0.5). `do_crease_or_blob_brush` squares it.
const BLOB_CREASE_PINCH_FACTOR: f64 = 0.5;

/// The positions a dab's falloff distances are measured from.
#[derive(Clone, Copy)]
enum Reference {
    /// The current positions (`position_data.eval`), as Draw and Blob read them.
    Current,
    /// The stroke-start positions (`orig_position_data_get_mesh`), as Draw Sharp reads them.
    StrokeStart,
}

/// Apply one dab of an offset brush.
pub(super) fn apply(field: &mut Field, dab: DabBrush, input: &DabInput) {
    match dab {
        DabBrush::Draw => area_offset_dab(field, input, Reference::Current),
        DabBrush::DrawSharp => area_offset_dab(field, input, Reference::StrokeStart),
        DabBrush::Blob => blob_dab(field, input),
        DabBrush::Nudge => nudge_dab(field, input),
        _ => unreachable!("offset::apply only receives Draw, Draw Sharp, Blob, and Nudge"),
    }
}

/// Blender's `calc_area_normal_and_center` (`sculpt.cc` 2185) for one dab: the smooth-weighted normal
/// of the vertices within `normal_radius_factor * radius` of the center. Draw, Draw Sharp, Blob, and
/// Nudge all have `use_accumulate` off, so the stroke-start positions are read. Blender's zero normal
/// sum gives `float3(0)` (`calc_sculpt_normal`), which makes every offset zero.
fn area_normal(field: &Field, dab: &DabInput, normal_radius_factor: f64) -> DVec3 {
    let radius = dab.radius * normal_radius_factor;
    area_plane(field, dab.center, radius, radius, false).map_or(DVec3::ZERO, |area| area.normal)
}

/// Blender's `do_draw_brush` (`draw.cc` 185-200) and `do_draw_sharp_brush` (`draw_sharp.cc` 195-210):
/// every vertex moves by `sculpt_normal_symm * radius * scale * bstrength * factor`, where the factor
/// is the falloff of the distance to the dab center times `1 - mask`. `scale` is 1 because Potter
/// strokes are in mesh space. The sign of `bstrength` (Draw Sharp's `brush_flip`) comes from the
/// framework's `brush_strength`. The sculpt normal is recomputed for every dab (`update_sculpt_normal`,
/// `sculpt.cc` 2720), since the essentials assets leave `BRUSH_ORIGINAL_NORMAL` off. Mirror passes
/// flip the main pass's normal and center.
fn area_offset_dab(field: &mut Field, dab: &DabInput, reference: Reference) {
    let normal = area_normal(field, dab, DRAW_NORMAL_RADIUS_FACTOR);
    for sign in &field.passes {
        let center = dab.center * *sign;
        let offset = normal * *sign * (dab.radius * dab.strength);
        for (index, position) in field.positions.iter_mut().enumerate() {
            let measured_from = match reference {
                Reference::Current => *position,
                Reference::StrokeStart => field.original[index],
            };
            let distance = measured_from.distance(center);
            if distance >= dab.radius {
                continue;
            }
            let factor = field.falloff.weight(distance / dab.radius) * field.mask_factors[index];
            *position += offset * factor;
        }
    }
}

/// Blender's `do_blob_brush` (`crease.cc` 286, through `do_crease_or_blob_brush` with the strength
/// inverted). Each vertex is pulled toward the dab center by `strength * factor`, the pull is projected
/// off the sculpt normal (`project_translations`), and the Draw offset `normal * radius * bstrength *
/// factor` is added. `strength = -|bstrength| * crease_correction`, with `crease_correction =
/// pinch^2 / alpha^2` and `alpha` the brush strength before squaring (`crease.cc` 209-217).
fn blob_dab(field: &mut Field, dab: &DabInput) {
    let normal = area_normal(field, dab, DRAW_NORMAL_RADIUS_FACTOR);
    let mut crease_correction = BLOB_CREASE_PINCH_FACTOR * BLOB_CREASE_PINCH_FACTOR;
    if dab.brush_alpha > 0.0 {
        crease_correction /= dab.brush_alpha * dab.brush_alpha;
    }
    let strength = -dab.strength.abs() * crease_correction;
    for sign in &field.passes {
        let center = dab.center * *sign;
        let plane = normal * *sign;
        let offset = plane * (dab.radius * dab.strength);
        for (index, position) in field.positions.iter_mut().enumerate() {
            let distance = position.distance(center);
            if distance >= dab.radius {
                continue;
            }
            let factor = field.falloff.weight(distance / dab.radius) * field.mask_factors[index];
            let pull = project_off_plane((center - *position) * (factor * strength), plane);
            *position += pull + offset * factor;
        }
    }
}

/// Blender's `project_translations` (`sculpt.cc` 7705): removes the component of `translation` along
/// `plane`. A zero plane leaves the translation unchanged, as Blender's length guard does.
fn project_off_plane(translation: DVec3, plane: DVec3) -> DVec3 {
    let length_squared = plane.length_squared();
    if length_squared == 0.0 {
        translation
    } else {
        translation - plane * (translation.dot(plane) / length_squared)
    }
}

/// Blender's `do_nudge_brush` (`draw.cc` 202-215): `offset = cross(cross(sculpt_normal_symm,
/// grab_delta_symm), sculpt_normal_symm) * bstrength`, with no radius or scale, applied through
/// `offset_positions` with the same falloff and mask factors as Draw. The grab delta of a mirror pass
/// is flipped with the normal, as `grab_delta_symm` is (`sculpt.cc` 3683).
fn nudge_dab(field: &mut Field, dab: &DabInput) {
    let normal = area_normal(field, dab, NUDGE_NORMAL_RADIUS_FACTOR);
    for sign in &field.passes {
        let center = dab.center * *sign;
        let plane = normal * *sign;
        let offset = plane.cross(dab.grab * *sign).cross(plane) * dab.strength;
        for (index, position) in field.positions.iter_mut().enumerate() {
            let distance = position.distance(center);
            if distance >= dab.radius {
                continue;
            }
            let factor = field.falloff.weight(distance / dab.radius) * field.mask_factors[index];
            *position += offset * factor;
        }
    }
}
