//! Blender 5.2.2 Boundary Inflate (`sculpt_boundary.cc` `calc_inflate_mesh`, `do_inflate_brush`).
//!
//! Every reached vertex moves along its stroke-start normal by its falloff factor times the dab's
//! amount. Like Bend, the offset is taken from the stroke-start position, so a dab replaces the one
//! before it. Inflate precomputes nothing: the normals and the factors come from the stroke.

use super::Boundary;
use crate::geom::sculpt_dabs::Field;

/// Blender's `calc_inflate_mesh` for one dab: each reached vertex is placed at its stroke-start
/// position plus its stroke-start normal times `factor * amount` (`calc_inflate_position`). A vertex
/// whose scaled factor is zero keeps its current position, as `filter_translations` does.
pub(super) fn apply(field: &mut Field, boundary: &Boundary, amount: f64) {
    for (index, position) in field.positions.iter_mut().enumerate() {
        let scaled = boundary.edit.factor[index] * amount;
        if scaled == 0.0 {
            continue;
        }
        *position = field.original[index] + field.normals[index] * scaled;
    }
}
