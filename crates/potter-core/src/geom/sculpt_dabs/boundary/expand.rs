//! Blender 5.2.2 Boundary Expand (`sculpt_boundary.cc` `slide_data_init_mesh`, `calc_slide_mesh`,
//! `do_slide_brush`).
//!
//! Every propagated vertex slides along its loop's direction by the amount the dab derives from the
//! grab's displacement (`get_mesh_strength`) times its falloff factor. The slide is taken from the
//! stroke-start position, so a dab replaces the one before it.

use glam::DVec3;

use super::{Boundary, DeformData, EditInfo};
use crate::geom::sculpt_dabs::Field;

/// Expand's per-vertex slide direction, indexed by vertex position.
pub(super) struct Data {
    /// `slide.directions`: unit vectors, or zero where Blender's direction is zero.
    directions: Vec<DVec3>,
}

/// Blender's `slide_data_init_mesh`: the direction of each loop vertex, from the loop vertex toward
/// the vertex the propagation reached last. Every other reached vertex takes its loop vertex's
/// direction.
pub(super) fn init(field: &Field, boundary: &Boundary) -> Data {
    let positions = &field.original;
    let EditInfo {
        original, steps, ..
    } = &boundary.edit;
    let mut directions = vec![DVec3::ZERO; positions.len()];
    for (index, step) in steps.iter().enumerate() {
        if *step != Some(boundary.max_steps) {
            continue;
        }
        let Some(origin) = original[index] else {
            continue;
        };
        directions[origin] = (positions[origin] - positions[index]).normalize_or_zero();
    }
    for (index, origin) in original.iter().enumerate() {
        let Some(origin) = *origin else {
            continue;
        };
        if steps[index].is_none() {
            continue;
        }
        directions[index] = directions[origin];
    }
    Data { directions }
}

/// Blender's `calc_slide_mesh` for one dab: each reached vertex moves from its stroke-start position
/// along its direction by `factor * amount`. A vertex whose scaled factor is zero keeps its current
/// position, as `filter_translations` does.
pub(super) fn apply(field: &mut Field, boundary: &Boundary, amount: f64) {
    let DeformData::Expand(data) = &boundary.deform else {
        return;
    };
    for (index, position) in field.positions.iter_mut().enumerate() {
        let scaled = boundary.edit.factor[index] * amount;
        if scaled == 0.0 {
            continue;
        }
        *position = field.original[index] + data.directions[index] * scaled;
    }
}
