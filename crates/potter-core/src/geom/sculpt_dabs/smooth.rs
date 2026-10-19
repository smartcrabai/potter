//! Blender 5.2.2 Smooth dab (`brushes/smooth.cc`, Laplacian deform type: the DNA default
//! `smooth_deform_type`, which the essentials asset does not override).

use glam::DVec3;

use super::{DabInput, Field};
use crate::geom::sculpt::DabBrush;

/// Blender's `max_iterations` in `iteration_strengths` (`smooth.cc:36`).
const MAX_ITERATIONS: f64 = 4.0;

/// Blender 5.2.2 Smooth for one dab. Only `DabBrush::Smooth` belongs here.
pub(super) fn apply(field: &mut Field, dab: DabBrush, input: &DabInput) {
    match dab {
        DabBrush::Smooth => smooth_dab(field, input),
        other => unreachable!("smooth::apply does not handle {other:?}"),
    }
}

/// Blender's `iteration_strengths` (`smooth.cc:34`): the strength, clamped to `[0, 1]`, becomes
/// whole iterations of 1.0 followed by one iteration with the remainder. The remainder is
/// `MAX_ITERATIONS * strength - count`, which is Blender's `last`.
fn iteration_strengths(strength: f64) -> Vec<f64> {
    let mut remaining = strength.clamp(0.0, 1.0) * MAX_ITERATIONS;
    let mut strengths = Vec::new();
    while remaining >= 1.0 {
        strengths.push(1.0);
        remaining -= 1.0;
    }
    strengths.push(remaining);
    strengths
}

/// Blender's `do_smooth_brush` (`smooth.cc:239`) for one dab, run once per symmetry pass. Each pass
/// smooths around its mirrored center (`location_symm`), and the passes run in order against the
/// positions the earlier passes left behind. Within a pass, every iteration is a full smoothing step.
fn smooth_dab(field: &mut Field, dab: &DabInput) {
    for sign in &field.passes {
        let center = dab.center * *sign;
        for strength in iteration_strengths(dab.strength) {
            let moves = smooth_moves(field, center, dab.radius, strength);
            for (index, translation) in moves {
                field.positions[index] += translation;
            }
        }
    }
}

/// One Blender smoothing iteration (`do_smooth_brush_mesh`, `smooth.cc:113-162`). The factors come
/// from the current positions (`calc_factors_common_mesh_indexed` reads `position_data.eval`), and
/// every neighbor average reads the same snapshot, since Blender computes all new positions before
/// it applies any translation. A vertex moves by `factor * (average - position)`, where the factor
/// is the falloff times the mask factor times the iteration strength (`scale_factors`,
/// `scale_translations`). Vertices outside the radius and vertices without neighbors do not move.
fn smooth_moves(field: &Field, center: DVec3, radius: f64, strength: f64) -> Vec<(usize, DVec3)> {
    field
        .positions
        .iter()
        .enumerate()
        .filter_map(|(index, position)| {
            let distance = position.distance(center);
            if distance >= radius {
                return None;
            }
            let factor =
                field.falloff.weight(distance / radius) * field.mask_factors[index] * strength;
            if factor == 0.0 {
                return None;
            }
            let average = neighbor_average(&field.positions, &field.neighbors[index])?;
            Some((index, (average - *position) * factor))
        })
        .collect()
}

/// Blender's `neighbor_data_average_mesh_check_loose` (`sculpt_smooth.cc:38`) with `mix_indices`
/// (`BKE_attribute_math.hh:380`): the mean of the neighbors, accumulated as `position * (1 / count)`.
/// Returns `None` for a vertex without neighbors, which Blender leaves where it is.
fn neighbor_average(positions: &[DVec3], neighbors: &[usize]) -> Option<DVec3> {
    if neighbors.is_empty() {
        return None;
    }
    let weight = 1.0 / neighbors.len() as f64;
    let mut average = DVec3::ZERO;
    for &neighbor in neighbors {
        average += positions[neighbor] * weight;
    }
    Some(average)
}
