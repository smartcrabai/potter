// SPDX-License-Identifier: GPL-3.0-or-later

//! Blender's `interp_weights_poly_v2` (`blenlib/intern/math_geom.cc`, Blender 5.2.2): mean-value
//! weights of a point inside a 2D polygon, with snapping to a vertex or an edge.
//!
//! Contract, matching Blender's float inputs:
//! - `polygon` vertices and `point` are f32 coordinates in one plane. Blender projects 3D corners
//!   onto the face plane in float before calling this; the caller owns that projection.
//! - The weights are f32, one per polygon vertex, in polygon order. They sum to one unless the
//!   point snaps to a vertex or edge, in which case only those two entries are non-zero.

/// How the mean-value walk ended: through the interior, or snapped to a vertex or an edge.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Snap {
    Interior,
    Point,
    Edge,
}

/// Blender's `Double2_Len`: the direction from the point to a polygon vertex, computed in double
/// from float coordinates, with its length.
type DirectionLength = ([f64; 2], f64);

/// Writes the mean-value weights of `point` in `polygon` into `weights`.
///
/// Returns `false`, leaving `weights` untouched, when `polygon` has fewer than three vertices or
/// `weights.len() != polygon.len()`. A point that lies on a vertex or edge within Blender's scale-aware
/// epsilon (`16 * FLT_EPSILON` times the largest coordinate difference) takes that vertex or
/// edge's linear weights instead of the mean-value weights.
pub(crate) fn interp_weights_poly_v2(
    weights: &mut [f32],
    polygon: &[[f32; 2]],
    point: [f32; 2],
) -> bool {
    let count = polygon.len();
    if count < 3 || weights.len() != count {
        return false;
    }
    let mut max_value = 0.0_f32;
    for vertex in polygon {
        max_value = max_value.max((vertex[0] - point[0]).abs());
        max_value = max_value.max((vertex[1] - point[1]).abs());
    }
    let epsilon = 16.0 * f32::EPSILON * max_value;
    let epsilon_squared = epsilon * epsilon;

    let mut total_weight = 0.0_f32;
    let mut snap = Snap::Interior;
    let mut current_index = count - 1;
    let mut next_index = 0;
    let mut current = polygon[current_index];
    let mut next = polygon[next_index];
    let mut current_direction = direction_to_point(polygon[count - 2], point);
    let mut next_direction = direction_to_point(current, point);
    let mut previous_half_tangent = half_tangent(current_direction, next_direction);
    while next_index < count {
        // Mayer et al.'s mean-value walk is unstable near the boundary, so snap there instead.
        if next_direction.1 < f64::from(epsilon) {
            snap = Snap::Point;
            break;
        }
        if line_segment_distance_squared(point, current, next) < epsilon_squared {
            snap = Snap::Edge;
            break;
        }
        current_direction = next_direction;
        next_direction = direction_to_point(next, point);
        let tangent = half_tangent(current_direction, next_direction);
        weights[current_index] = if current_direction.1 == 0.0 {
            0.0
        } else {
            ((previous_half_tangent + tangent) / current_direction.1) as f32
        };
        total_weight += weights[current_index];
        current_index = next_index;
        next_index += 1;
        current = next;
        if next_index < count {
            next = polygon[next_index];
        }
        previous_half_tangent = tangent;
    }

    match snap {
        Snap::Interior => {
            if total_weight != 0.0 {
                for weight in weights.iter_mut() {
                    *weight /= total_weight;
                }
            }
        }
        Snap::Point => {
            weights.fill(0.0);
            weights[current_index] = 1.0;
        }
        Snap::Edge => {
            weights.fill(0.0);
            let factor = line_point_factor(point, current, next).clamp(0.0, 1.0);
            weights[current_index] = 1.0 - factor;
            weights[next_index] = factor;
        }
    }
    true
}

/// Blender's `sub_v2db_v2fl_v2fl` and `len_v2_db`: `vertex - point` in double, and its length.
fn direction_to_point(vertex: [f32; 2], point: [f32; 2]) -> DirectionLength {
    let direction = [
        f64::from(vertex[0]) - f64::from(point[0]),
        f64::from(vertex[1]) - f64::from(point[1]),
    ];
    let length = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
    (direction, length)
}

/// Blender's `mean_value_half_tan_v2_db`: `(len_a * len_b - dot) / cross`, or zero for a degenerate
/// or non-finite result.
fn half_tangent(current: DirectionLength, next: DirectionLength) -> f64 {
    let area = current.0[0] * next.0[1] - current.0[1] * next.0[0];
    if area == 0.0 {
        return 0.0;
    }
    let dot = current.0[0] * next.0[0] + current.0[1] * next.0[1];
    let result = (current.1 * next.1 - dot) / area;
    if result.is_finite() { result } else { 0.0 }
}

/// Blender's `dist_squared_to_line_segment_v2` in float: squared distance from `point` to the
/// segment `first`-`second`, with the closest point clamped to the segment.
fn line_segment_distance_squared(point: [f32; 2], first: [f32; 2], second: [f32; 2]) -> f32 {
    let factor = line_point_factor(point, first, second);
    let closest = if factor <= 0.0 {
        first
    } else if factor >= 1.0 {
        second
    } else {
        [
            first[0] + (second[0] - first[0]) * factor,
            first[1] + (second[1] - first[1]) * factor,
        ]
    };
    let difference = [closest[0] - point[0], closest[1] - point[1]];
    difference[0] * difference[0] + difference[1] * difference[1]
}

/// Blender's `line_point_factor_v2`: the projection parameter of `point` on the line `first`-`second`,
/// or zero when the line has no length.
fn line_point_factor(point: [f32; 2], first: [f32; 2], second: [f32; 2]) -> f32 {
    let direction = [second[0] - first[0], second[1] - first[1]];
    let offset = [point[0] - first[0], point[1] - first[1]];
    let denominator = direction[0] * direction[0] + direction[1] * direction[1];
    if denominator == 0.0 {
        0.0
    } else {
        (direction[0] * offset[0] + direction[1] * offset[1]) / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::interp_weights_poly_v2;

    const SQUARE: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    fn weights_at(point: [f32; 2]) -> [f32; 4] {
        let mut weights = [0.0_f32; 4];
        assert!(interp_weights_poly_v2(&mut weights, &SQUARE, point));
        weights
    }

    #[test]
    fn interior_centre_has_equal_mean_value_weights() {
        for weight in weights_at([0.5, 0.5]) {
            assert!((weight - 0.25).abs() < 1.0e-6);
        }
    }

    #[test]
    fn interior_weights_sum_to_one() {
        let total: f32 = weights_at([0.25, 0.4]).iter().sum();
        assert!((total - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn point_on_edge_snaps_to_its_two_vertices() {
        let weights = weights_at([0.5, 1.0e-6]);
        assert!((weights[0] - 0.5).abs() < 1.0e-6);
        assert!((weights[1] - 0.5).abs() < 1.0e-6);
        assert_eq!(weights[2], 0.0);
        assert_eq!(weights[3], 0.0);
    }

    #[test]
    fn point_on_vertex_snaps_to_that_vertex() {
        assert_eq!(weights_at([0.0, 0.0]), [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn degenerate_inputs_are_refused() {
        let mut weights = [7.0_f32; 2];
        assert!(!interp_weights_poly_v2(&mut weights, &SQUARE[..2], [0.5, 0.5]));
        assert_eq!(weights, [7.0, 7.0]);
        let mut short = [0.0_f32; 3];
        assert!(!interp_weights_poly_v2(&mut short, &SQUARE, [0.5, 0.5]));
    }
}
