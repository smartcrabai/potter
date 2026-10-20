//! Blender 5.2.2 Pose (`sculpt_pose.cc`) on the dab framework.
//!
//! The first dab builds an inverse-kinematics chain. Its origin is the centroid of the vertices
//! just outside the brush radius, found by a flood fill from the vertex nearest the stroke's first
//! dab that expands only through vertices inside the radius (`calc_pose_origin_and_factor_mesh`).
//! Mirror symmetry seeds the flood from the nearest vertex to each mirrored active vertex, and a
//! vertex is inside when it lies within the radius of the brush or of any mirrored brush location.
//! The chain's weights are 1 on the flood region and are blurred by Blender's four neighbor-average
//! passes (`pose_brush_init`). Each dab then sets the chain's transform toward the anchored grab
//! target (Squash & Stretch scales the segment without solving it) and moves every vertex by
//! `mask * weight * (T(orig) - orig)`, with `T` the segment transform about the origin (`calc_mesh`).
//! Positions come from the stroke-start mesh every dab, as Blender's `orig_position_data` does for
//! Pose.
//!
//! Potter's chain has one segment and Blender's defaults for the settings it does not expose:
//! `pose_ik_segments` 1, `pose_origin_type` TOPOLOGY, `use_connected_only` on, `pose_offset` 0,
//! `use_pose_ik_anchored` on, `use_pose_lock_rotation` off, `pose_smooth_iterations` 4. Each
//! symmetry area gets its own transform, with the chain mirrored into that area (`do_pose_brush`,
//! `area_transform`). The inverted (twist) branch is not supported: the twist branch's roll reads
//! screen pixels.

use std::collections::VecDeque;

use glam::{DMat3, DQuat, DVec3};

use super::{DabInput, Field, grab, nearest_in_sphere, nearest_vertex, pivot_symmetry};
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

/// Blender's `do_pose_brush` for one dab. The chain is built by the stroke's first dab, and each dab
/// sets its transform toward the anchored grab target. Blender runs the brush once for all enabled
/// mirror axes (its mirror passes other than 0 return early), so each vertex moves by the transform
/// of its own symmetry area.
pub(super) fn apply(field: &mut Field, deform: PoseDeform, input: &DabInput) {
    if field.positions.is_empty() {
        return;
    }
    let chain = match field.pose.chain.take() {
        Some(chain) => chain,
        None => init_chain(field, input),
    };
    // `ik_target = cache->location + grab_delta * bstrength` (`calc_rotate_deform`,
    // `calc_scale_deform`, `calc_squash_stretch_deform`). The topology origin has no
    // `grab_delta_offset`.
    let target = input.center + input.grab * input.strength;
    // Squash & Stretch scales the chain without solving it (`calc_squash_stretch_deform`), so its
    // rotation stays the identity.
    let (rotation, scale) = match deform {
        PoseDeform::RotateTwist => (anchored_rotation(&chain, target), DVec3::ONE),
        PoseDeform::ScaleTranslate => (
            anchored_rotation(&chain, target),
            DVec3::splat(scale_from_grab(&chain, target)),
        ),
        PoseDeform::SquashStretch => (DQuat::IDENTITY, squash_scale(&chain, target)),
    };
    let aligned = deform == PoseDeform::SquashStretch;
    let rotation = DMat3::from_quat(rotation);
    let mirror = field.mirror;
    let areas: [(DVec3, DMat3); 8] = std::array::from_fn(|mask| {
        area_transform(&chain, rotation, scale, aligned, flip_signs(mask))
    });
    for (index, position) in field.positions.iter_mut().enumerate() {
        let original = field.original[index];
        let (pivot, transform) = areas[flip_mask(original, input.center, mirror)];
        let moved = pivot + transform * (original - pivot);
        let weight = chain.weights[index] * field.mask_factors[index];
        *position = original + (moved - original) * weight;
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
    // Blender's `find_symm_verts_mesh` (`sculpt_expand.cc` 574) with the brush radius: the nearest
    // vertex to each mirrored active vertex, when the mesh's bounds lie within the radius of it.
    // These seeds start the flood without being marked visited.
    let mut queue = VecDeque::from([active]);
    for pass in field.passes.iter().skip(1) {
        let mirrored = positions[active] * *pass;
        if let Some(nearest) = nearest_in_sphere(positions, mirrored, input.radius) {
            queue.push_back(nearest);
        }
    }
    let mirror = field.mirror;
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
            if field
                .passes
                .iter()
                .any(|pass| (center * *pass).distance(co) < input.radius)
            {
                queue.push_back(to);
            } else if pivot_symmetry(co, center, mirror) {
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
            // Blender's `mix_indices` over no neighbors is zero (`BKE_attribute_math.hh` 380), so an
            // isolated vertex's weight falls to zero rather than holding its value.
            weights[index] = if neighbors.is_empty() {
                0.0
            } else {
                let sum: f64 = neighbors.iter().map(|&neighbor| previous[neighbor]).sum();
                sum / neighbors.len() as f64
            };
        }
    }
    Chain {
        origin,
        head: center,
        len: center.distance(origin),
        weights,
    }
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

/// `calc_scale_from_grab_delta`: the scale that stretches the segment so its head reaches the
/// target's distance along the segment direction.
fn scale_from_grab(chain: &Chain, target: DVec3) -> f64 {
    let direction = (chain.head - chain.origin).normalize_or_zero();
    let along = direction.dot(target - chain.head);
    if chain.len == 0.0 {
        // An isolated active vertex gives a zero-length segment, which has no direction to stretch.
        return 1.0;
    }
    chain.len / (chain.len - along)
}

/// Blender's `calc_squash_stretch_deform`: the stretch along the segment is
/// `calc_scale_from_grab_delta`, and x and y scale by `sqrt(1 / |stretch|)` with the stretch's sign,
/// which keeps the volume. A stretch under `1e-5` collapses the segment.
fn squash_scale(chain: &Chain, target: DVec3) -> DVec3 {
    let stretch = scale_from_grab(chain, target);
    if stretch.abs() < 1e-5 {
        return DVec3::ZERO;
    }
    let across = (1.0 / stretch.abs()).sqrt() * stretch.signum();
    DVec3::new(across, across, stretch)
}

/// Blender's `do_pose_brush` transform for one symmetry area, as its pivot and the matrix applied
/// about that pivot. `flips` mirrors the chain into the area (`flip_v3_by_symm_area` and
/// `flip_quat_by_symm_area`): the origin is mirrored, and a mirrored rotation is conjugated by the
/// same flip. `orig` equals `initial_orig` after the anchored solve, so Blender's translation
/// between them is zero and omitted.
fn area_transform(
    chain: &Chain,
    rotation: DMat3,
    scale: DVec3,
    aligned: bool,
    flips: DVec3,
) -> (DVec3, DMat3) {
    let pivot = flips * chain.origin;
    if aligned {
        // Squash & Stretch scales x and y by the same factor, so its transform is the isotropic
        // part plus the stretch along the flipped segment direction `n`: `a·I + (s − a)·n·nᵀ`. This
        // equals Blender's `frame · diag(a, a, s) · frameᵀ` for any orthonormal x, y about `n`
        // (checked to 2e-15 over random directions), so the frame is not built.
        let normal = (flips * (chain.head - chain.origin)).normalize_or_zero();
        let across = scale.x;
        let outer = DMat3::from_cols(normal * normal.x, normal * normal.y, normal * normal.z);
        let transform = DMat3::from_diagonal(DVec3::splat(across)) + (scale.z - across) * outer;
        return (pivot, transform);
    }
    let mirrored = DMat3::from_diagonal(flips) * rotation * DMat3::from_diagonal(flips);
    (pivot, mirrored * DMat3::from_diagonal(scale))
}

/// The symmetry area of `point` as a flip mask (X = 1, Y = 2, Z = 4). An enabled axis flips when
/// `point` and `pivot` lie on opposite sides of it: Blender's `flip_v3_by_symm_area` flips for the
/// vertex's area and again for a negative pivot, and the two cancel when both agree.
fn flip_mask(point: DVec3, pivot: DVec3, mirror: DVec3) -> usize {
    (0..3)
        .filter(|&axis| mirror[axis] < 0.0 && (point[axis] < 0.0) != (pivot[axis] < 0.0))
        .fold(0, |mask, axis| mask | (1 << axis))
}

/// The mirror signs of a flip mask: -1 on each flipped axis, as Blender's `symmetry_flip`.
fn flip_signs(mask: usize) -> DVec3 {
    let sign = |bit: usize| if mask & bit != 0 { -1.0 } else { 1.0 };
    DVec3::new(sign(1), sign(2), sign(4))
}
