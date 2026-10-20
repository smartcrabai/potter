//! Blender 5.2.2 Boundary Smooth (`sculpt_boundary.cc` `do_smooth_brush`, `calc_smooth_mesh`,
//! `calc_average_position`, `calc_smooth_position`).
//!
//! Every propagated vertex moves toward the average of its ring neighbors that sit on the same
//! propagation step, by its scaled factor. The average reads the positions the dab starts from, so
//! all vertices move together: one Jacobi step per dab. A vertex with no same-step neighbor keeps
//! its position.

use glam::DVec3;

use super::Boundary;
use crate::geom::sculpt_dabs::Field;

/// Blender's `do_smooth_brush` for one dab. `amount` is the strength (`get_mesh_strength`).
pub(super) fn apply(field: &mut Field, boundary: &Boundary, amount: f64) {
    let start = field.positions.clone();
    let steps = &boundary.edit.steps;
    for (index, position) in field.positions.iter_mut().enumerate() {
        let scaled = boundary.edit.factor[index] * amount;
        if scaled == 0.0 {
            continue;
        }
        let Some(step) = steps[index] else {
            continue;
        };
        let (sum, count) = field.topology.ring[index]
            .iter()
            .filter(|&&neighbor| steps[neighbor] == Some(step))
            .fold((DVec3::ZERO, 0_u32), |(sum, count), &neighbor| {
                (sum + start[neighbor], count + 1)
            });
        if count == 0 {
            continue;
        }
        let average = sum / f64::from(count);
        *position = start[index] + (average - start[index]) * scaled;
    }
}
