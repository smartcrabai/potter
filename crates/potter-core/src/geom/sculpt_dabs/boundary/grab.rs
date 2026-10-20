//! Blender 5.2.2 Boundary Grab (`sculpt_boundary.cc` `calc_grab_mesh`): every propagated vertex
//! moves by the grab delta times its falloff factor and the strength, from its stroke-start
//! position. A vertex whose scaled factor is zero keeps its current position.

use super::Boundary;
use crate::geom::sculpt_dabs::{DabInput, Field};

/// Blender's `do_grab_brush` for one dab. `amount` is the strength (`get_mesh_strength`).
pub(super) fn apply(field: &mut Field, boundary: &Boundary, input: &DabInput, amount: f64) {
    for (index, position) in field.positions.iter_mut().enumerate() {
        let scaled = boundary.edit.factor[index] * amount;
        if scaled == 0.0 {
            continue;
        }
        *position = field.original[index] + input.grab * scaled;
    }
}
