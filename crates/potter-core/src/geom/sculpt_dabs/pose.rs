//! Blender 5.2.2 Pose (`sculpt_pose.cc`) on the dab framework.
//!
//! The first dab builds an inverse-kinematics chain. Its origin is the centroid of the vertices
//! just outside the brush radius, found by a flood fill from the vertex nearest the stroke's first
//! dab that expands only through vertices inside the radius (`calc_pose_origin_and_factor_mesh`).
//! The chain's weights are 1 on the flood region and are blurred by Blender's four neighbor-average
//! passes (`pose_brush_init`). Each dab then solves the chain toward the anchored grab target and
//! moves every vertex by `mask * weight * (T(orig) - orig)`, with `T` the segment transform about
//! the origin (`calc_mesh`). Positions come from the stroke-start mesh every dab, as Blender's
//! `orig_position_data` does for Pose.
//!
//! Potter's chain has one segment and Blender's defaults for the settings it does not expose:
//! `pose_ik_segments` 1, `pose_origin_type` TOPOLOGY, `use_connected_only` on, `pose_offset` 0,
//! `use_pose_ik_anchored` on, `use_pose_lock_rotation` off, `pose_smooth_iterations` 4. Symmetry
//! and the inverted (twist) branch are not supported: the twist branch's roll reads screen pixels.

use std::collections::VecDeque;

use glam::{DQuat, DVec3};

use super::{DabInput, Field, grab};
use crate::geom::sculpt::PoseDeform;

/// `pose_smooth_iterations` default: Blender blurs each segment's weights this many times.
const SMOOTH_ITERATIONS: usize = 4;

/// Pose state that persists between dabs: the chain, built by the stroke's first dab.
#[derive(Default)]
pub(super) struct State {
    chain: Option<Chain>,
}

/// Blender's `IKChain` with one segment. `origin` is both `initial_orig` and the anchor the
/// anchored solve returns to, so the chain never moves between dabs.
struct Chain {
    /// `segment.initial_orig`: the pose origin, the pivot of the segment's transform.
    origin: DVec3,
    /// `segment.initial_head`: the first dab's center, `cache->location`.
    head: DVec3,
    /// `segment.len`: the distance from the origin to the head.
    len: f64,
    /// `segment.weights`, after `blur_geometry_data_array`, indexed like the field's vertices.
    weights: Vec<f64>,
}

pub(super) fn apply(field: &mut Field, deform: PoseDeform, input: &DabInput) {
    if field.positions.is_empty() {
        return;
    }
    let chain = match field.pose.chain.take() {
        Some(chain) => chain,
        None => init_chain(field, input),
    };
    // `ik_target = cache->location + grab_delta * bstrength` (`calc_rotate_deform`,
    // `calc_scale_deform`). The topology origin has no `grab_delta_offset`.
    let target = input.center + input.grab * input.strength;
    let rotation = anchored_rotation(&chain, target);
    let scale = match deform {
        PoseDeform::RotateTwist => 1.0,
        PoseDeform::ScaleTranslate => scale_from_grab(&chain, target),
    };
    let pivot = chain.origin;
    let transform = |point: DVec3| pivot + rotation.mul_vec3((point - pivot) * scale);
    for (index, position) in field.positions.iter_mut().enumerate() {
        let original = field.original[index];
        let weight = chain.weights[index] * field.mask_factors[index];
        *position = original + (transform(original) - original) * weight;
    }
    field.pose.chain = Some(chain);
}

/// Blender's `pose_brush_init` and `ik_chain_init_topology` for one segment: the origin from the
/// topology flood fill, the weights from the flood region blurred `SMOOTH_ITERATIONS` times.
fn init_chain(field: &Field, input: &DabInput) -> Chain {
    let center = input.center;
    let positions = &field.original;
    let active = nearest_vertex(positions, center);
    let mut factor = vec![0.0_f64; positions.len()];
    factor[active] = 1.0;
    let mut visited = vec![false; positions.len()];
    let mut queue = VecDeque::from([active]);
    let mut outside_sum = DVec3::ZERO;
    let mut outside_count = 0_usize;
    // Blender's `fallback_floodfill_origin` starts at the brush location and keeps the visited
    // vertex farthest from it.
    let mut fallback = center;
    while let Some(from) = queue.pop_front() {
        for &to in &field.topology.ring[from] {
            if visited[to] {
                continue;
            }
            visited[to] = true;
            factor[to] = 1.0;
            let co = positions[to];
            if center.distance_squared(fallback) < center.distance_squared(co) {
                fallback = co;
            }
            if co.distance(center) < input.radius {
                queue.push_back(to);
            } else {
                outside_sum += co;
                outside_count += 1;
            }
        }
    }
    let origin = if outside_count > 0 {
        outside_sum / outside_count as f64
    } else {
        fallback
    };
    let mut weights = factor;
    for _ in 0..SMOOTH_ITERATIONS {
        let previous = weights.clone();
        for (index, neighbors) in field.topology.ring.iter().enumerate() {
            if neighbors.is_empty() {
                continue;
            }
            let sum: f64 = neighbors.iter().map(|&neighbor| previous[neighbor]).sum();
            weights[index] = sum / neighbors.len() as f64;
        }
    }
    Chain {
        origin,
        head: center,
        len: center.distance(origin),
        weights,
    }
}

/// The vertex nearest `point`, the lowest index winning ties. Blender takes the active vertex from
/// the ray cast under the cursor, which is this vertex when the stroke starts on the mesh.
fn nearest_vertex(positions: &[DVec3], point: DVec3) -> usize {
    let mut best = 0;
    let mut best_distance = f64::INFINITY;
    for (index, position) in positions.iter().enumerate() {
        let distance = position.distance_squared(point);
        if distance < best_distance {
            best = index;
            best_distance = distance;
        }
    }
    best
}

/// `solve_ik_chain` for the single anchored segment: the rotation that turns the segment's initial
/// direction toward the target, about the origin. The anchored solve returns the origin to its
/// initial place, so only the rotation is kept.
fn anchored_rotation(chain: &Chain, target: DVec3) -> DQuat {
    let initial = (chain.head - chain.origin).normalize_or_zero();
    let current = (target - chain.origin).normalize_or_zero();
    if initial == DVec3::ZERO || current == DVec3::ZERO {
        return DQuat::IDENTITY;
    }
    let (axis, angle) = grab::between(initial, current);
    DQuat::from_axis_angle(axis, angle)
}

/// `calc_scale_from_grab_delta`: the uniform scale that stretches the segment so its head reaches
/// the target's distance along the segment direction.
fn scale_from_grab(chain: &Chain, target: DVec3) -> f64 {
    let direction = (chain.head - chain.origin).normalize_or_zero();
    let along = direction.dot(target - chain.head);
    chain.len / (chain.len - along)
}
