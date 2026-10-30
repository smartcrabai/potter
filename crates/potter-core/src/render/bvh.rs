use glam::DVec3;

use crate::error::{ErrorCode, PotError, Result};
use crate::model::RayKind;
use crate::render::raster::Triangle;

const LEAF_TRIANGLE_COUNT: usize = 4;
const BARYCENTRIC_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BvhHit {
    pub triangle_index: usize,
    pub distance: f64,
    pub barycentric: [f64; 3],
}

/// The ray a [`Bvh::walk_hits`] walk follows: where it starts, its direction, the offset it restarts past each
/// hit, the distance left along it, and its kind: the triangles whose object hides that kind are passed through. A
/// `None` kind is a local probe of the objects themselves (Cycles' `scene_intersect_local`), which takes every triangle.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WalkRay {
    pub origin: DVec3,
    pub direction: DVec3,
    pub epsilon: f64,
    pub remaining: f64,
    pub kind: Option<RayKind>,
}

/// A BVH ray query: the ray, its extent, whether back faces count, and which triangles it may hit. Traversal
/// carries it as one value instead of a long argument list.
#[derive(Clone, Copy)]
struct RayQuery<'a> {
    origin: DVec3,
    direction: DVec3,
    t_min: f64,
    t_max: f64,
    two_sided: bool,
    /// Admits a triangle by its object's ray mask (`Triangle::ray_mask`).
    accepts: &'a dyn Fn(u8) -> bool,
}

#[derive(Debug)]
pub(crate) struct Bvh {
    nodes: Vec<Node>,
    triangle_indices: Vec<usize>,
    /// Parallel to `triangle_indices`: leaf tests read this contiguous array and never the source triangles.
    leaf_triangles: Vec<LeafTriangle>,
    root: Option<usize>,
    triangle_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct LeafTriangle {
    positions: [DVec3; 3],
    triangle_index: usize,
    double_sided: bool,
    ray_mask: u8,
}

/// One primitive handed to [`Bvh::build_primitives`]: its corners, its sidedness and the ray kinds its object admits.
#[derive(Debug, Clone, Copy)]
struct Primitive {
    positions: [DVec3; 3],
    double_sided: bool,
    ray_mask: u8,
}

#[derive(Debug, Clone, Copy)]
struct ChildEntry {
    node: usize,
    span: (f64, f64),
    near: f64,
}

#[derive(Debug, Clone, Copy)]
struct Node {
    bounds: Aabb,
    kind: NodeKind,
}

#[derive(Debug, Clone, Copy)]
enum NodeKind {
    Leaf { start: usize, count: usize },
    Branch { left: usize, right: usize },
}

#[derive(Debug, Clone, Copy)]
struct Aabb {
    min: DVec3,
    max: DVec3,
}

impl Bvh {
    pub(crate) fn build(triangles: &[Triangle]) -> Result<Self> {
        for triangle in triangles {
            if triangle
                .positions
                .iter()
                .any(|point| !is_finite_vec3(*point))
            {
                return Err(PotError::invalid_argument(
                    "triangle positions must be finite to build a BVH",
                ));
            }
        }
        Self::build_primitives(triangles.iter().map(|triangle| Primitive {
            positions: triangle.positions,
            double_sided: triangle.double_sided,
            ray_mask: triangle.ray_mask,
        }))
    }

    /// A BVH over triangles given by their corner positions; primitive `i` is the `i`th item. These primitives carry
    /// no double-sided flag, so they are one-sided, and no ray mask, so every ray kind meets them.
    pub(crate) fn build_from_positions(
        positions: impl IntoIterator<Item = [DVec3; 3]>,
    ) -> Result<Self> {
        Self::build_primitives(positions.into_iter().map(|corners| Primitive {
            positions: corners,
            double_sided: false,
            ray_mask: RayKind::ALL_MASK,
        }))
    }

    fn build_primitives(items: impl IntoIterator<Item = Primitive>) -> Result<Self> {
        let primitives: Vec<LeafTriangle> = items
            .into_iter()
            .enumerate()
            .map(|(triangle_index, primitive)| LeafTriangle {
                positions: primitive.positions,
                triangle_index,
                double_sided: primitive.double_sided,
                ray_mask: primitive.ray_mask,
            })
            .collect();
        let primitive_bounds: Vec<Aabb> = primitives
            .iter()
            .map(|primitive| Aabb::for_corners(&primitive.positions))
            .collect();
        let triangle_count = primitive_bounds.len();
        let node_capacity = maximum_node_count(triangle_count)?;

        let mut triangle_indices = Vec::new();
        triangle_indices
            .try_reserve_exact(triangle_count)
            .map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "BVH triangle count exceeds available memory",
                )
            })?;
        triangle_indices.extend(0..triangle_count);

        let mut nodes = Vec::new();
        nodes.try_reserve_exact(node_capacity).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "BVH node count exceeds available memory",
            )
        })?;

        let root = if triangle_count == 0 {
            None
        } else {
            Some(build_node(
                &primitive_bounds,
                &mut triangle_indices,
                0,
                &mut nodes,
            )?)
        };

        let mut leaf_triangles = Vec::new();
        leaf_triangles
            .try_reserve_exact(triangle_count)
            .map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "BVH triangle count exceeds available memory",
                )
            })?;
        for &index in &triangle_indices {
            let Some(primitive) = primitives.get(index) else {
                return Err(PotError::new(
                    ErrorCode::InternalError,
                    "BVH triangle index is out of range",
                ));
            };
            leaf_triangles.push(*primitive);
        }

        Ok(Self {
            nodes,
            triangle_indices,
            leaf_triangles,
            root,
            triangle_count,
        })
    }

    /// The nearest hit among the triangles `accepts` admits, given each triangle's object ray mask, so a ray passes
    /// through the others. Back faces count.
    pub(crate) fn closest_hit_where(
        &self,
        triangles: &[Triangle],
        origin: DVec3,
        direction: DVec3,
        t_min: f64,
        t_max: f64,
        accepts: &dyn Fn(u8) -> bool,
    ) -> Option<BvhHit> {
        let query = RayQuery {
            origin,
            direction,
            t_min,
            t_max,
            two_sided: true,
            accepts,
        };
        self.closest_hit_mode(triangles, query)
    }

    /// Whether the ray hits any triangle admitted by `accepts`. Back faces count.
    pub(crate) fn any_hit_where(
        &self,
        triangles: &[Triangle],
        origin: DVec3,
        direction: DVec3,
        t_min: f64,
        t_max: f64,
        accepts: &dyn Fn(u8) -> bool,
    ) -> bool {
        self.any_hit_mode(
            triangles,
            RayQuery {
                origin,
                direction,
                t_min,
                t_max,
                two_sided: true,
                accepts,
            },
        )
    }

    /// Walks the surfaces `walk` meets, nearest first, while `walk.remaining` is left along it. `visit` receives each
    /// hit and the origin the ray left from, and returns whether the walk goes on. The ray restarts `walk.epsilon`
    /// past each hit, and the walk ends when the distance is spent. Triangles whose object is hidden from the ray's
    /// kind are passed through. Returns the origin and the distance where the walk stopped.
    pub(crate) fn walk_hits(
        &self,
        triangles: &[Triangle],
        walk: WalkRay,
        mut visit: impl FnMut(BvhHit, DVec3) -> bool,
    ) -> (DVec3, f64) {
        let WalkRay {
            mut origin,
            direction,
            epsilon,
            mut remaining,
            kind,
        } = walk;
        while let Some(hit) =
            self.closest_hit_where(triangles, origin, direction, epsilon, remaining, &|mask| {
                kind.is_none_or(|kind| mask & kind.bit() != 0)
            })
        {
            if !visit(hit, origin) {
                break;
            }
            let advance = hit.distance + epsilon;
            origin += direction * advance;
            if remaining.is_finite() {
                remaining = (remaining - advance).max(0.0);
                if remaining <= epsilon {
                    break;
                }
            }
        }
        (origin, remaining)
    }

    /// Clears `out` and fills it with the index of every primitive whose leaf bounds the ray
    /// `origin + t * direction` (t >= 0) may cross. Every primitive the ray hits is among them.
    pub(crate) fn overlapping(&self, origin: DVec3, direction: DVec3, out: &mut Vec<usize>) {
        out.clear();
        if let Some(root) = self.root {
            self.collect(root, origin, direction, out);
        }
    }

    fn collect(&self, node_index: usize, origin: DVec3, direction: DVec3, out: &mut Vec<usize>) {
        let Some(node) = self.nodes.get(node_index).copied() else {
            return;
        };
        if node
            .bounds
            .padded(origin)
            .slab_span(origin, direction)
            .and_then(|span| clip_span(span, 0.0, f64::INFINITY))
            .is_none()
        {
            return;
        }
        match node.kind {
            NodeKind::Leaf { start, count } => {
                if let Some(indices) = start
                    .checked_add(count)
                    .and_then(|end| self.triangle_indices.get(start..end))
                {
                    out.extend_from_slice(indices);
                }
            }
            NodeKind::Branch { left, right } => {
                self.collect(left, origin, direction, out);
                self.collect(right, origin, direction, out);
            }
        }
    }

    fn closest_hit_mode(&self, triangles: &[Triangle], query: RayQuery<'_>) -> Option<BvhHit> {
        if triangles.len() != self.triangle_count
            || !is_finite_vec3(query.origin)
            || !is_finite_vec3(query.direction)
            || query.t_min.is_nan()
            || query.t_max.is_nan()
            || query.t_min > query.t_max
        {
            return None;
        }

        let mut nearest = None;
        if let Some(root) = self.root
            && let Some(span) = self
                .nodes
                .get(root)
                .and_then(|node| node.bounds.slab_span(query.origin, query.direction))
        {
            self.visit(root, span, &query, &mut nearest);
        }
        nearest
    }

    fn any_hit_mode(&self, triangles: &[Triangle], query: RayQuery<'_>) -> bool {
        if triangles.len() != self.triangle_count
            || !is_finite_vec3(query.origin)
            || !is_finite_vec3(query.direction)
            || query.t_min.is_nan()
            || query.t_max.is_nan()
            || query.t_min > query.t_max
        {
            return false;
        }

        let Some(root) = self.root else {
            return false;
        };
        let Some(span) = self
            .nodes
            .get(root)
            .and_then(|node| node.bounds.slab_span(query.origin, query.direction))
        else {
            return false;
        };
        self.any_visit(root, span, &query)
    }

    fn any_visit(&self, node_index: usize, span: (f64, f64), query: &RayQuery<'_>) -> bool {
        let Some(node) = self.nodes.get(node_index).copied() else {
            return false;
        };
        if clip_span(span, query.t_min, query.t_max).is_none() {
            return false;
        }

        match node.kind {
            NodeKind::Leaf { start, count } => {
                let Some(end) = start.checked_add(count) else {
                    return false;
                };
                for offset in start..end {
                    let Some(leaf) = self.leaf_triangles.get(offset) else {
                        continue;
                    };
                    if !(query.accepts)(leaf.ray_mask) {
                        continue;
                    }
                    if intersect_triangle(
                        leaf,
                        query.origin,
                        query.direction,
                        query.t_min,
                        query.t_max,
                        query.two_sided,
                    )
                    .is_some()
                    {
                        return true;
                    }
                }
                false
            }
            NodeKind::Branch { left, right } => {
                let left_entry = self.entry(left, query, query.t_max);
                let right_entry = self.entry(right, query, query.t_max);
                match (left_entry, right_entry) {
                    (Some(left_entry), Some(right_entry)) => {
                        let (first, second) = if left_entry.near <= right_entry.near {
                            (left_entry, right_entry)
                        } else {
                            (right_entry, left_entry)
                        };
                        self.any_visit(first.node, first.span, query)
                            || self.any_visit(second.node, second.span, query)
                    }
                    (Some(entry), None) | (None, Some(entry)) => {
                        self.any_visit(entry.node, entry.span, query)
                    }
                    (None, None) => false,
                }
            }
        }
    }

    fn entry(&self, index: usize, query: &RayQuery<'_>, max_distance: f64) -> Option<ChildEntry> {
        let span = self
            .nodes
            .get(index)?
            .bounds
            .slab_span(query.origin, query.direction)?;
        let (near, _) = clip_span(span, query.t_min, max_distance)?;
        Some(ChildEntry {
            node: index,
            span,
            near,
        })
    }

    fn visit(
        &self,
        node_index: usize,
        span: (f64, f64),
        query: &RayQuery<'_>,
        nearest: &mut Option<BvhHit>,
    ) {
        let Some(node) = self.nodes.get(node_index).copied() else {
            return;
        };
        let current_max = nearest.map_or(query.t_max, |hit: BvhHit| hit.distance.min(query.t_max));
        if clip_span(span, query.t_min, current_max).is_none() {
            return;
        }

        match node.kind {
            NodeKind::Leaf { start, count } => {
                let Some(end) = start.checked_add(count) else {
                    return;
                };
                for offset in start..end {
                    let Some(leaf) = self.leaf_triangles.get(offset) else {
                        continue;
                    };
                    if !(query.accepts)(leaf.ray_mask) {
                        continue;
                    }
                    let Some(hit) = intersect_triangle(
                        leaf,
                        query.origin,
                        query.direction,
                        query.t_min,
                        current_max,
                        query.two_sided,
                    ) else {
                        continue;
                    };
                    if nearest.is_none_or(|previous| {
                        hit.distance < previous.distance
                            || (crate::float::equal_f64(hit.distance, previous.distance)
                                && hit.triangle_index < previous.triangle_index)
                    }) {
                        *nearest = Some(hit);
                    }
                }
            }
            NodeKind::Branch { left, right } => {
                let left_entry = self.entry(left, query, current_max);
                let right_entry = self.entry(right, query, current_max);
                match (left_entry, right_entry) {
                    (Some(left_entry), Some(right_entry)) => {
                        let (first, second) = if left_entry.near <= right_entry.near {
                            (left_entry, right_entry)
                        } else {
                            (right_entry, left_entry)
                        };
                        self.visit(first.node, first.span, query, nearest);
                        self.visit(second.node, second.span, query, nearest);
                    }
                    (Some(entry), None) | (None, Some(entry)) => {
                        self.visit(entry.node, entry.span, query, nearest);
                    }
                    (None, None) => {}
                }
            }
        }
    }
}

fn maximum_node_count(triangle_count: usize) -> Result<usize> {
    if triangle_count == 0 {
        return Ok(0);
    }
    let rounded_count = triangle_count
        .checked_add(LEAF_TRIANGLE_COUNT - 1)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "BVH triangle count overflow"))?;
    let leaf_count = rounded_count / LEAF_TRIANGLE_COUNT;
    leaf_count
        .checked_mul(2)
        .and_then(|count| count.checked_sub(1))
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "BVH node count overflow"))
}

fn build_node(
    primitive_bounds: &[Aabb],
    triangle_indices: &mut [usize],
    start: usize,
    nodes: &mut Vec<Node>,
) -> Result<usize> {
    let bounds = bounds_for_indices(primitive_bounds, triangle_indices)?;
    if triangle_indices.len() <= LEAF_TRIANGLE_COUNT {
        let index = nodes.len();
        let _next_count = index
            .checked_add(1)
            .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "BVH node index overflow"))?;
        let end = start
            .checked_add(triangle_indices.len())
            .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "BVH leaf range overflow"))?;
        let count = end
            .checked_sub(start)
            .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "BVH leaf range overflow"))?;
        nodes.push(Node {
            bounds,
            kind: NodeKind::Leaf { start, count },
        });
        return Ok(index);
    }

    let axis = bounds.largest_axis();
    triangle_indices.sort_unstable_by(|left, right| {
        let left_center = primitive_bounds[*left].center(axis);
        let right_center = primitive_bounds[*right].center(axis);
        left_center
            .total_cmp(&right_center)
            .then_with(|| left.cmp(right))
    });

    let middle = triangle_indices.len() / 2;
    let right_start = start
        .checked_add(middle)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "BVH split index overflow"))?;
    let (left_indices, right_indices) = triangle_indices.split_at_mut(middle);
    let left = build_node(primitive_bounds, left_indices, start, nodes)?;
    let right = build_node(primitive_bounds, right_indices, right_start, nodes)?;
    let index = nodes.len();
    index
        .checked_add(1)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "BVH node index overflow"))?;
    nodes.push(Node {
        bounds,
        kind: NodeKind::Branch { left, right },
    });
    Ok(index)
}

fn bounds_for_indices(primitive_bounds: &[Aabb], indices: &[usize]) -> Result<Aabb> {
    let Some(first_index) = indices.first().copied() else {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "cannot bound an empty BVH node",
        ));
    };
    let Some(mut bounds) = primitive_bounds.get(first_index).copied() else {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "BVH triangle index is out of range",
        ));
    };
    for &index in &indices[1..] {
        let Some(other) = primitive_bounds.get(index).copied() else {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "BVH triangle index is out of range",
            ));
        };
        bounds.include(other);
    }
    Ok(bounds)
}

impl Aabb {
    fn for_corners(corners: &[DVec3; 3]) -> Self {
        let mut min = corners[0];
        let mut max = corners[0];
        for point in &corners[1..] {
            min = min.min(*point);
            max = max.max(*point);
        }
        Self { min, max }
    }

    fn include(&mut self, other: Self) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    fn center(self, axis: usize) -> f64 {
        self.min[axis] * 0.5 + self.max[axis] * 0.5
    }

    fn largest_axis(self) -> usize {
        let extent = self.max - self.min;
        let mut axis = 0;
        if extent.y > extent.x {
            axis = 1;
        }
        if extent.z > extent[axis] {
            axis = 2;
        }
        axis
    }

    /// The ray's span through the box's slabs, before clamping to a query interval, so it is the same for every
    /// query. `None` when the slabs are empty or a ray parallel to an axis starts outside the box.
    fn slab_span(self, origin: DVec3, direction: DVec3) -> Option<(f64, f64)> {
        let mut near = f64::NEG_INFINITY;
        let mut far = f64::INFINITY;
        for axis in 0..3 {
            let ray_origin = origin[axis];
            let ray_direction = direction[axis];
            if ray_direction == 0.0 {
                if ray_origin < self.min[axis] || ray_origin > self.max[axis] {
                    return None;
                }
                continue;
            }

            let mut first = (self.min[axis] - ray_origin) / ray_direction;
            let mut second = (self.max[axis] - ray_origin) / ray_direction;
            if first > second {
                std::mem::swap(&mut first, &mut second);
            }
            near = near.max(first);
            far = far.min(second);
            if near > far {
                return None;
            }
        }
        Some((near, far))
    }

    /// The box grown by a margin far above the rounding error of the slab test at `origin`, so a
    /// primitive that a ray crosses is never culled by rounding.
    fn padded(self, origin: DVec3) -> Self {
        let scale = self
            .min
            .abs()
            .max(self.max.abs())
            .max(origin.abs())
            .max_element();
        let margin = DVec3::splat(1.0e-9 * (1.0 + scale));
        Self {
            min: self.min - margin,
            max: self.max + margin,
        }
    }
}

/// Clamps a box's slab span to the query interval `[t_min, t_max]`; `None` when they are disjoint.
fn clip_span(span: (f64, f64), t_min: f64, t_max: f64) -> Option<(f64, f64)> {
    let (slab_near, slab_far) = span;
    let near = t_min.max(slab_near);
    let far = t_max.min(slab_far);
    if near > far {
        return None;
    }
    Some((near, far))
}

fn intersect_triangle(
    leaf: &LeafTriangle,
    origin: DVec3,
    direction: DVec3,
    t_min: f64,
    t_max: f64,
    two_sided: bool,
) -> Option<BvhHit> {
    let edge1 = leaf.positions[1] - leaf.positions[0];
    let edge2 = leaf.positions[2] - leaf.positions[0];
    let p_vector = direction.cross(edge2);
    let determinant = edge1.dot(p_vector);
    if !determinant.is_finite()
        || determinant.abs() <= f64::MIN_POSITIVE
        || (!two_sided && !leaf.double_sided && determinant <= 0.0)
    {
        return None;
    }

    let inverse_determinant = 1.0 / determinant;
    if !inverse_determinant.is_finite() {
        return None;
    }
    let from_vertex = origin - leaf.positions[0];
    let barycentric_1 = from_vertex.dot(p_vector) * inverse_determinant;
    let q_vector = from_vertex.cross(edge1);
    let barycentric_2 = direction.dot(q_vector) * inverse_determinant;
    let distance = edge2.dot(q_vector) * inverse_determinant;
    let barycentric_0 = 1.0 - barycentric_1 - barycentric_2;
    if !barycentric_0.is_finite()
        || !barycentric_1.is_finite()
        || !barycentric_2.is_finite()
        || !distance.is_finite()
        || distance < t_min
        || distance > t_max
        || barycentric_0 < -BARYCENTRIC_EPSILON
        || barycentric_1 < -BARYCENTRIC_EPSILON
        || barycentric_2 < -BARYCENTRIC_EPSILON
        || barycentric_0 > 1.0 + BARYCENTRIC_EPSILON
        || barycentric_1 > 1.0 + BARYCENTRIC_EPSILON
        || barycentric_2 > 1.0 + BARYCENTRIC_EPSILON
    {
        return None;
    }

    Some(BvhHit {
        triangle_index: leaf.triangle_index,
        distance,
        barycentric: [
            barycentric_0.clamp(0.0, 1.0),
            barycentric_1.clamp(0.0, 1.0),
            barycentric_2.clamp(0.0, 1.0),
        ],
    })
}

fn is_finite_vec3(vector: DVec3) -> bool {
    vector.x.is_finite() && vector.y.is_finite() && vector.z.is_finite()
}

#[cfg(test)]
mod tests {
    use glam::DVec3;
    use proptest::prelude::*;

    use crate::model::{RayKind, RayVisibility};
    use crate::render::raster::{AlphaMode, Triangle};

    use super::{Bvh, BvhHit, RayQuery};

    proptest! {
        #[test]
        fn bvh_hit_queries_match_brute_force_moller_trumbore(
            triangle_data in prop::collection::vec(
                (-8.0_f64..8.0, -8.0_f64..8.0, -8.0_f64..8.0, 0.1_f64..2.0, any::<bool>(), any::<bool>()),
                1..=24,
            ),
            target_index in any::<usize>(),
            barycentric_1 in 0.05_f64..0.45,
            barycentric_2 in 0.05_f64..0.45,
            direction_x in -1.5_f64..1.5,
            direction_y in -1.5_f64..1.5,
            direction_z in 0.1_f64..1.5,
            ray_distance in 2.0_f64..8.0,
            t_min in 0.0_f64..1.0,
            t_max in 20.0_f64..40.0,
        ) {
            let target_index = target_index % triangle_data.len();
            let triangles = triangle_data
                .iter()
                .enumerate()
                .map(|(index, (x, y, z, size, double_sided, shadow_visible))| {
                    let center = DVec3::new(*x, *y, *z);
                    Triangle {
                        positions: [
                            center + DVec3::new(-*size, -*size, 0.0),
                            center + DVec3::new(*size, -*size, 0.0),
                            center + DVec3::new(0.0, *size, 0.0),
                        ],
                        generated: crate::geom::GeneratedCorners::default(),
                        geometry_vertex_slots: [0; 3],
                        geometry_vertex_weights: [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        object_index: index as u32,
                        element_index: index as u32,
                        color: [1.0; 4],
                        alpha_mode: AlphaMode::Opaque,
                        alpha_threshold: 0.5,
                        double_sided: index == target_index || *double_sided,
                        material_index: 0,
                        uv: [[0.0; 2]; 3],
                        layer_uvs: Vec::new(),
                        layer_tangents: Vec::new(),
                        normals: None,
                        layer_colors: Vec::new(),
                        lit_per_pixel: false,
                        ray_mask: if *shadow_visible {
                            RayVisibility::default().mask()
                        } else {
                            RayVisibility::default().mask() & !RayKind::Shadow.bit()
                        },
                    }
                })
                .collect::<Vec<_>>();

            let target = &triangles[target_index];
            let barycentric_0 = 1.0 - barycentric_1 - barycentric_2;
            let target_point = target.positions[0] * barycentric_0
                + target.positions[1] * barycentric_1
                + target.positions[2] * barycentric_2;
            let direction = DVec3::new(direction_x, direction_y, direction_z);
            let origin = target_point + direction * ray_distance;
            let ray_direction = -direction;

            let bvh = Bvh::build(&triangles);
            prop_assert!(bvh.is_ok(), "BVH build failed: {bvh:?}");
            let Ok(bvh) = bvh else { return Ok(()); };
            let actual = bvh.closest_hit_mode(
                &triangles,
                RayQuery {
                    origin,
                    direction: ray_direction,
                    t_min,
                    t_max,
                    two_sided: false,
                    accepts: &|_| true,
                },
            );
            let expected = brute_force_hit(&triangles, origin, ray_direction, t_min, t_max, false);
            prop_assert!(expected.is_some(), "constructed ray should hit its target triangle");
            prop_assert_eq!(actual, expected);
            let actual_two_sided =
                bvh.closest_hit_where(&triangles, origin, ray_direction, t_min, t_max, &|_| true);
            let expected_two_sided =
                brute_force_hit(&triangles, origin, ray_direction, t_min, t_max, true);
            prop_assert_eq!(actual_two_sided, expected_two_sided);
            let actual_any =
                bvh.any_hit_where(&triangles, origin, ray_direction, t_min, t_max, &|_| true);
            prop_assert_eq!(actual_any, expected_two_sided.is_some());
            let accepts_shadow = |mask| mask & RayKind::Shadow.bit() != 0;
            let actual_shadow =
                bvh.closest_hit_where(&triangles, origin, ray_direction, t_min, t_max, &accepts_shadow);
            let actual_any_shadow =
                bvh.any_hit_where(&triangles, origin, ray_direction, t_min, t_max, &accepts_shadow);
            prop_assert_eq!(actual_any_shadow, actual_shadow.is_some());
            prop_assert!(!bvh.any_hit_where(
                &triangles,
                origin,
                ray_direction,
                t_min,
                t_max,
                &|_| false,
            ));
        }
    }

    fn brute_force_hit(
        triangles: &[Triangle],
        origin: DVec3,
        direction: DVec3,
        t_min: f64,
        t_max: f64,
        two_sided: bool,
    ) -> Option<BvhHit> {
        let mut closest = None;
        for (triangle_index, triangle) in triangles.iter().enumerate() {
            let edge1 = triangle.positions[1] - triangle.positions[0];
            let edge2 = triangle.positions[2] - triangle.positions[0];
            let p_vector = direction.cross(edge2);
            let determinant = edge1.dot(p_vector);
            if !determinant.is_finite()
                || determinant.abs() <= f64::MIN_POSITIVE
                || (!two_sided && !triangle.double_sided && determinant <= 0.0)
            {
                continue;
            }

            let inverse_determinant = 1.0 / determinant;
            if !inverse_determinant.is_finite() {
                continue;
            }
            let from_vertex = origin - triangle.positions[0];
            let barycentric_1 = from_vertex.dot(p_vector) * inverse_determinant;
            let q_vector = from_vertex.cross(edge1);
            let barycentric_2 = direction.dot(q_vector) * inverse_determinant;
            let distance = edge2.dot(q_vector) * inverse_determinant;
            let barycentric_0 = 1.0 - barycentric_1 - barycentric_2;
            if !barycentric_0.is_finite()
                || !barycentric_1.is_finite()
                || !barycentric_2.is_finite()
                || !distance.is_finite()
                || distance < t_min
                || distance > t_max
                || barycentric_0 < -1.0e-12
                || barycentric_1 < -1.0e-12
                || barycentric_2 < -1.0e-12
                || barycentric_0 > 1.0 + 1.0e-12
                || barycentric_1 > 1.0 + 1.0e-12
                || barycentric_2 > 1.0 + 1.0e-12
            {
                continue;
            }

            let hit = BvhHit {
                triangle_index,
                distance,
                barycentric: [
                    barycentric_0.clamp(0.0, 1.0),
                    barycentric_1.clamp(0.0, 1.0),
                    barycentric_2.clamp(0.0, 1.0),
                ],
            };
            if closest.is_none_or(|previous: BvhHit| {
                hit.distance < previous.distance
                    || (hit.distance == previous.distance
                        && hit.triangle_index < previous.triangle_index)
            }) {
                closest = Some(hit);
            }
        }
        closest
    }
}
