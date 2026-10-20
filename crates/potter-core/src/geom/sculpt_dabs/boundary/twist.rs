//! Blender 5.2.2 Boundary Twist (`sculpt_boundary.cc` `populate_twist_data`, `calc_twist_mesh`).
//!
//! Every propagated vertex rotates about the loop's centroid by its falloff factor times the angle
//! the dab derives from the grab's displacement. The rotation is taken from the stroke-start
//! position, so a dab replaces the one before it.

use glam::DVec3;

use super::{Boundary, DeformData, rotate};
use crate::geom::sculpt_dabs::Field;

/// Twist's rotation center and axis, computed once per stroke from the loop.
pub(super) struct Data {
    /// `twist.pivot_position`: the mean of the loop's vertices, duplicates included.
    center: DVec3,
    /// `twist.rotation_axis`: unit length, from the initial vertex toward `boundary.pivot`.
    axis: DVec3,
}

/// Blender's `twist_data_init_mesh` and `populate_twist_data`. The axis points from the initial
/// vertex toward the propagation pivot, not toward the centroid: Blender's `populate_twist_data`
/// subtracts `boundary.pivot_position`, so the axis is `pivot - initial` even though the rotation
/// is about the centroid. A zero difference gives a zero axis.
pub(super) fn init(field: &Field, boundary: &Boundary) -> Data {
    let sum: DVec3 = boundary
        .verts
        .iter()
        .map(|&vertex| field.original[vertex])
        .sum();
    let center = sum / boundary.verts.len() as f64;
    let axis = (boundary.pivot - field.original[boundary.initial]).normalize_or_zero();
    Data { center, axis }
}

/// Blender's `calc_twist_mesh` for one dab: each reached vertex rotates its stroke-start position
/// about the center by `factor * angle`. A vertex whose scaled factor is zero keeps its current
/// position, as `filter_translations` does.
pub(super) fn apply(field: &mut Field, boundary: &Boundary, angle: f64) {
    let DeformData::Twist(data) = &boundary.deform else {
        return;
    };
    for (index, position) in field.positions.iter_mut().enumerate() {
        let scaled = boundary.edit.factor[index] * angle;
        if scaled == 0.0 {
            continue;
        }
        *position = rotate(field.original[index] - data.center, data.axis, scaled) + data.center;
    }
}
