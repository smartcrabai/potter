//! Blender 5.2.2 Boundary Bend (`sculpt_boundary.cc` `bend_data_init_mesh`, `calc_bend_mesh`).
//!
//! Every propagated vertex rotates about the axis of its loop vertex by the amount the dab derives
//! from the grab's displacement (`get_mesh_strength`) times its falloff factor. The rotation is
//! taken from the stroke-start position, so a dab replaces the one before it.

use glam::DVec3;

use super::{Boundary, DeformData, EditInfo, rotate};
use crate::geom::sculpt_dabs::Field;

/// Bend's per-vertex pivot and axis, indexed by vertex position (`bend_data_init_mesh`).
pub(super) struct Data {
    /// `bend.pivot_positions`.
    pivot: Vec<DVec3>,
    /// `bend.pivot_rotation_axis`: unit axes, or zero where Blender's axis is zero.
    axis: Vec<DVec3>,
}

/// Blender's `bend_data_init_mesh`: the pivot and axis of each vertex the propagation reached.
/// A vertex at the last step defines its loop vertex's pivot as the step's position, with the axis
/// perpendicular to the loop-to-vertex direction and the vertex normal. Every other reached vertex
/// takes its loop vertex's pivot and axis.
pub(super) fn init(field: &Field, boundary: &Boundary) -> Data {
    let positions = &field.original;
    let count = positions.len();
    let EditInfo {
        original, steps, ..
    } = &boundary.edit;
    let mut pivot = vec![DVec3::ZERO; count];
    let mut axis = vec![DVec3::ZERO; count];
    for (index, step) in steps.iter().enumerate() {
        if *step != Some(boundary.max_steps) {
            continue;
        }
        let Some(origin) = original[index] else {
            continue;
        };
        let direction = positions[origin] - positions[index];
        axis[origin] = direction.cross(field.normals[index]).normalize_or_zero();
        pivot[origin] = positions[index];
    }
    for (index, origin) in original.iter().enumerate() {
        let Some(origin) = *origin else {
            continue;
        };
        if steps[index].is_none() {
            continue;
        }
        pivot[index] = pivot[origin];
        axis[index] = axis[origin];
    }
    Data { pivot, axis }
}

/// Blender's `calc_bend_mesh` for one dab: each reached vertex rotates its stroke-start position by
/// `factor * angle` about its pivot. A vertex whose scaled factor is zero keeps its current
/// position, as `filter_translations` does.
pub(super) fn apply(field: &mut Field, boundary: &Boundary, angle: f64) {
    let DeformData::Bend(data) = &boundary.deform else {
        return;
    };
    for (index, position) in field.positions.iter_mut().enumerate() {
        let scaled = boundary.edit.factor[index] * angle;
        if scaled == 0.0 {
            continue;
        }
        let from_pivot = field.original[index] - data.pivot[index];
        *position = rotate(from_pivot, data.axis[index], scaled) + data.pivot[index];
    }
}
