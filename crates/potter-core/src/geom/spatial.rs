// SPDX-License-Identifier: GPL-3.0-or-later

//! Conservative BVH traversal over primitive bounds. Callers retain ownership of primitive data and
//! perform their exact nearest-point or ray-intersection tests only for candidates.

use glam::DVec3;

use crate::error::{ErrorCode, PotError, Result};

const LEAF_SIZE: usize = 4;

#[derive(Clone, Copy)]
struct Bounds {
    min: DVec3,
    max: DVec3,
}

struct Node {
    bounds: Bounds,
    kind: NodeKind,
}

#[derive(Clone, Copy)]
enum NodeKind {
    Leaf { start: usize, end: usize },
    Branch { left: usize, right: usize },
}

/// A median-split BVH whose primitive payload remains in the caller's storage.
pub(crate) struct BoundsBvh {
    bounds: Vec<Bounds>,
    order: Vec<usize>,
    nodes: Vec<Node>,
}

impl BoundsBvh {
    /// Build from one inclusive minimum/maximum pair per primitive.
    pub(crate) fn new(bounds: &[(DVec3, DVec3)]) -> Result<Self> {
        let capacity = if bounds.is_empty() {
            0
        } else {
            bounds
                .len()
                .checked_mul(2)
                .and_then(|count| count.checked_sub(1))
                .ok_or_else(bvh_memory_error)?
        };
        let mut primitive_bounds = Vec::new();
        primitive_bounds
            .try_reserve_exact(bounds.len())
            .map_err(|_| bvh_memory_error())?;
        for &(min, max) in bounds {
            if !min.is_finite() || !max.is_finite() || min.cmpgt(max).any() {
                return Err(PotError::invalid_argument(
                    "spatial primitive bounds must be finite and ordered",
                ));
            }
            primitive_bounds.push(Bounds { min, max });
        }
        let mut order = Vec::new();
        order
            .try_reserve_exact(bounds.len())
            .map_err(|_| bvh_memory_error())?;
        order.extend(0..bounds.len());
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(capacity)
            .map_err(|_| bvh_memory_error())?;
        if !order.is_empty() {
            let end = order.len();
            build_node(&primitive_bounds, &mut order, &mut nodes, 0, end);
        }
        Ok(Self {
            bounds: primitive_bounds,
            order,
            nodes,
        })
    }

    /// Visit only bounds whose squared distance can improve the best exact result.
    pub(crate) fn nearest<T>(
        &self,
        point: DVec3,
        max_distance_squared: f64,
        mut test: impl FnMut(usize, f64) -> Option<(f64, T)>,
    ) -> Option<(usize, f64, T)> {
        if !point.is_finite() || max_distance_squared.is_nan() || max_distance_squared < 0.0 {
            return None;
        }
        let mut nearest = None;
        let mut best_distance_squared = max_distance_squared;
        if !self.nodes.is_empty() {
            self.visit_nearest(
                0,
                point,
                &mut best_distance_squared,
                &mut test,
                &mut nearest,
            );
        }
        nearest
    }

    fn visit_nearest<T>(
        &self,
        node_index: usize,
        point: DVec3,
        best_distance_squared: &mut f64,
        test: &mut impl FnMut(usize, f64) -> Option<(f64, T)>,
        nearest: &mut Option<(usize, f64, T)>,
    ) {
        let node = &self.nodes[node_index];
        if node.bounds.distance_squared(point) > *best_distance_squared {
            return;
        }
        match node.kind {
            NodeKind::Leaf { start, end } => {
                for offset in start..end {
                    let primitive = self.order[offset];
                    if self.bounds[primitive].distance_squared(point) > *best_distance_squared {
                        continue;
                    }
                    let Some((distance_squared, payload)) = test(primitive, *best_distance_squared)
                    else {
                        continue;
                    };
                    if !distance_squared.is_finite()
                        || distance_squared < 0.0
                        || distance_squared > *best_distance_squared
                    {
                        continue;
                    }
                    if nearest
                        .as_ref()
                        .is_none_or(|(old_primitive, old_distance, _)| {
                            let ordering = distance_squared.total_cmp(old_distance);
                            ordering.is_lt() || (ordering.is_eq() && primitive < *old_primitive)
                        })
                    {
                        *best_distance_squared = distance_squared;
                        *nearest = Some((primitive, distance_squared, payload));
                    }
                }
            }
            NodeKind::Branch { left, right } => {
                let left_distance = self.nodes[left].bounds.distance_squared(point);
                let right_distance = self.nodes[right].bounds.distance_squared(point);
                let (first, first_distance, second, second_distance) =
                    if left_distance <= right_distance {
                        (left, left_distance, right, right_distance)
                    } else {
                        (right, right_distance, left, left_distance)
                    };
                if first_distance <= *best_distance_squared {
                    self.visit_nearest(first, point, best_distance_squared, test, nearest);
                }
                if second_distance <= *best_distance_squared {
                    self.visit_nearest(second, point, best_distance_squared, test, nearest);
                }
            }
        }
    }

    /// Find the nearest exact ray hit, pruning only bounds that cannot intersect the current interval.
    pub(crate) fn raycast<T>(
        &self,
        origin: DVec3,
        direction: DVec3,
        t_min: f64,
        t_max: f64,
        mut test: impl FnMut(usize, f64, f64) -> Option<(f64, T)>,
    ) -> Option<(usize, f64, T)> {
        if !origin.is_finite()
            || !direction.is_finite()
            || direction == DVec3::ZERO
            || t_min.is_nan()
            || t_max.is_nan()
            || t_min > t_max
        {
            return None;
        }
        let mut nearest = None;
        if !self.nodes.is_empty() {
            self.visit_raycast(0, origin, direction, t_min, t_max, &mut test, &mut nearest);
        }
        nearest
    }

    fn visit_raycast<T>(
        &self,
        node_index: usize,
        origin: DVec3,
        direction: DVec3,
        t_min: f64,
        t_max: f64,
        test: &mut impl FnMut(usize, f64, f64) -> Option<(f64, T)>,
        nearest: &mut Option<(usize, f64, T)>,
    ) {
        let node = &self.nodes[node_index];
        if node
            .bounds
            .ray_interval(
                origin,
                direction,
                t_min,
                current_ray_max(t_max, nearest.as_ref()),
            )
            .is_none()
        {
            return;
        }
        match node.kind {
            NodeKind::Leaf { start, end } => {
                for offset in start..end {
                    let primitive = self.order[offset];
                    let current_max = current_ray_max(t_max, nearest.as_ref());
                    if self.bounds[primitive]
                        .ray_interval(origin, direction, t_min, current_max)
                        .is_none()
                    {
                        continue;
                    }
                    let Some((distance, payload)) = test(primitive, t_min, current_max) else {
                        continue;
                    };
                    if !distance.is_finite() || distance < t_min || distance > current_max {
                        continue;
                    }
                    if nearest
                        .as_ref()
                        .is_none_or(|(old_primitive, old_distance, _)| {
                            let ordering = distance.total_cmp(old_distance);
                            ordering.is_lt() || (ordering.is_eq() && primitive < *old_primitive)
                        })
                    {
                        *nearest = Some((primitive, distance, payload));
                    }
                }
            }
            NodeKind::Branch { left, right } => {
                let current_max = current_ray_max(t_max, nearest.as_ref());
                let left_interval =
                    self.nodes[left]
                        .bounds
                        .ray_interval(origin, direction, t_min, current_max);
                let right_interval =
                    self.nodes[right]
                        .bounds
                        .ray_interval(origin, direction, t_min, current_max);
                match (left_interval, right_interval) {
                    (Some(left_interval), Some(right_interval))
                        if left_interval.0 <= right_interval.0 =>
                    {
                        self.visit_raycast(left, origin, direction, t_min, t_max, test, nearest);
                        self.visit_raycast(right, origin, direction, t_min, t_max, test, nearest);
                    }
                    (Some(_), Some(_)) => {
                        self.visit_raycast(right, origin, direction, t_min, t_max, test, nearest);
                        self.visit_raycast(left, origin, direction, t_min, t_max, test, nearest);
                    }
                    (Some(_), None) => {
                        self.visit_raycast(left, origin, direction, t_min, t_max, test, nearest);
                    }
                    (None, Some(_)) => {
                        self.visit_raycast(right, origin, direction, t_min, t_max, test, nearest);
                    }
                    (None, None) => {}
                }
            }
        }
    }
}

fn current_ray_max<T>(maximum: f64, nearest: Option<&(usize, f64, T)>) -> f64 {
    nearest.map_or(maximum, |(_, distance, _)| maximum.min(*distance))
}

fn build_node(
    bounds: &[Bounds],
    order: &mut [usize],
    nodes: &mut Vec<Node>,
    start: usize,
    end: usize,
) -> usize {
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    let mut center_min = DVec3::splat(f64::INFINITY);
    let mut center_max = DVec3::splat(f64::NEG_INFINITY);
    for &primitive in &order[start..end] {
        let bound = bounds[primitive];
        min = min.min(bound.min);
        max = max.max(bound.max);
        let center = (bound.min + bound.max) * 0.5;
        center_min = center_min.min(center);
        center_max = center_max.max(center);
    }
    let node_index = nodes.len();
    nodes.push(Node {
        bounds: Bounds { min, max },
        kind: NodeKind::Leaf { start, end },
    });
    if end - start > LEAF_SIZE {
        let extent = center_max - center_min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        let middle = start + (end - start) / 2;
        order[start..end].select_nth_unstable_by(middle - start, |left, right| {
            let left_center = f64::midpoint(bounds[*left].min[axis], bounds[*left].max[axis]);
            let right_center = f64::midpoint(bounds[*right].min[axis], bounds[*right].max[axis]);
            left_center
                .total_cmp(&right_center)
                .then_with(|| left.cmp(right))
        });
        let left = build_node(bounds, order, nodes, start, middle);
        let right = build_node(bounds, order, nodes, middle, end);
        nodes[node_index].kind = NodeKind::Branch { left, right };
    }
    node_index
}

impl Bounds {
    fn distance_squared(self, point: DVec3) -> f64 {
        let delta = (self.min - point).max(DVec3::ZERO) + (self.max - point).min(DVec3::ZERO);
        delta.length_squared()
    }

    fn ray_interval(
        self,
        origin: DVec3,
        direction: DVec3,
        mut t_min: f64,
        mut t_max: f64,
    ) -> Option<(f64, f64)> {
        for axis in 0..3 {
            if direction[axis] == 0.0 {
                if origin[axis] < self.min[axis] || origin[axis] > self.max[axis] {
                    return None;
                }
                continue;
            }
            let inverse = direction[axis].recip();
            let first = (self.min[axis] - origin[axis]) * inverse;
            let second = (self.max[axis] - origin[axis]) * inverse;
            t_min = t_min.max(first.min(second));
            t_max = t_max.min(first.max(second));
            if t_min > t_max {
                return None;
            }
        }
        Some((t_min, t_max))
    }
}

fn bvh_memory_error() -> PotError {
    PotError::new(
        ErrorCode::LimitExceeded,
        "spatial BVH exceeds available memory",
    )
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::BoundsBvh;

    trait TestValue<T> {
        fn test_value(self) -> T;
    }

    impl<T, E: std::fmt::Debug> TestValue<T> for Result<T, E> {
        fn test_value(self) -> T {
            match self {
                Ok(value) => value,
                Err(error) => panic!("invalid spatial BVH test fixture: {error:?}"),
            }
        }
    }

    impl<T> TestValue<T> for Option<T> {
        fn test_value(self) -> T {
            match self {
                Some(value) => value,
                None => panic!("spatial BVH test expected a value"),
            }
        }
    }

    #[test]
    fn nearest_visits_only_bounds_that_can_improve_the_exact_distance() {
        let mut boxes = vec![(DVec3::ZERO, DVec3::splat(0.1))];
        boxes.extend((1..9).map(|index| {
            let x = f64::from(index) * 10.0;
            (DVec3::new(x, 0.0, 0.0), DVec3::new(x + 0.1, 0.1, 0.1))
        }));
        let bvh = BoundsBvh::new(&boxes).test_value();
        let mut tested = Vec::new();
        let (index, distance, payload) = bvh
            .nearest(DVec3::new(0.05, 0.05, 1.0), f64::INFINITY, |index, _| {
                tested.push(index);
                Some((1.0, index * 2))
            })
            .test_value();
        assert_eq!((index, distance, payload), (0, 1.0, 0));
        assert_eq!(tested, vec![0]);
    }

    #[test]
    fn raycast_orders_hits_by_parameter_and_prunes_far_bounds() {
        let mut boxes = vec![
            (DVec3::new(0.0, 0.0, 1.0), DVec3::new(1.0, 1.0, 2.0)),
            (DVec3::new(0.0, 0.0, 4.0), DVec3::new(1.0, 1.0, 5.0)),
        ];
        boxes.extend((0..8).map(|index| {
            let x = 4.0 + f64::from(index) * 10.0;
            (DVec3::new(x, 0.0, 0.0), DVec3::new(x + 1.0, 1.0, 1.0))
        }));
        let bvh = BoundsBvh::new(&boxes).test_value();
        let mut tested = Vec::new();
        let (index, distance, payload) = bvh
            .raycast(
                DVec3::new(0.5, 0.5, 0.0),
                DVec3::Z,
                0.0,
                10.0,
                |index, _, _| {
                    tested.push(index);
                    Some((if index == 0 { 1.0 } else { 4.0 }, index))
                },
            )
            .test_value();
        assert_eq!((index, distance, payload), (0, 1.0, 0));
        assert!(!tested.contains(&1));
        assert!(!tested.contains(&2));
    }
}
