use glam::DVec3;

use crate::error::{ErrorCode, PotError, Result};
use crate::render::raster::Triangle;

const LEAF_TRIANGLE_COUNT: usize = 4;
const BARYCENTRIC_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BvhHit {
    pub triangle_index: usize,
    pub distance: f64,
    pub barycentric: [f64; 3],
}

#[derive(Debug)]
pub(crate) struct Bvh {
    nodes: Vec<Node>,
    triangle_indices: Vec<usize>,
    root: Option<usize>,
    triangle_count: usize,
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
        let triangle_count = triangles.len();
        let node_capacity = maximum_node_count(triangle_count)?;

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
            Some(build_node(triangles, &mut triangle_indices, 0, &mut nodes)?)
        };

        Ok(Self {
            nodes,
            triangle_indices,
            root,
            triangle_count,
        })
    }

    pub(crate) fn closest_hit_two_sided(
        &self,
        triangles: &[Triangle],
        origin: DVec3,
        direction: DVec3,
        t_min: f64,
        t_max: f64,
    ) -> Option<BvhHit> {
        self.closest_hit_mode(triangles, origin, direction, t_min, t_max, true)
    }

    fn closest_hit_mode(
        &self,
        triangles: &[Triangle],
        origin: DVec3,
        direction: DVec3,
        t_min: f64,
        t_max: f64,
        two_sided: bool,
    ) -> Option<BvhHit> {
        if triangles.len() != self.triangle_count
            || !is_finite_vec3(origin)
            || !is_finite_vec3(direction)
            || t_min.is_nan()
            || t_max.is_nan()
            || t_min > t_max
        {
            return None;
        }

        let mut nearest = None;
        if let Some(root) = self.root {
            self.visit(
                root,
                triangles,
                origin,
                direction,
                t_min,
                t_max,
                two_sided,
                &mut nearest,
            );
        }
        nearest
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "BVH traversal carries query bounds and sidedness through recursive nodes"
    )]
    fn visit(
        &self,
        node_index: usize,
        triangles: &[Triangle],
        origin: DVec3,
        direction: DVec3,
        t_min: f64,
        t_max: f64,
        two_sided: bool,
        nearest: &mut Option<BvhHit>,
    ) {
        let Some(node) = self.nodes.get(node_index).copied() else {
            return;
        };
        let current_max = nearest.map_or(t_max, |hit: BvhHit| hit.distance.min(t_max));
        if node
            .bounds
            .intersect_interval(origin, direction, t_min, current_max)
            .is_none()
        {
            return;
        }

        match node.kind {
            NodeKind::Leaf { start, count } => {
                let Some(end) = start.checked_add(count) else {
                    return;
                };
                for offset in start..end {
                    let Some(&triangle_index) = self.triangle_indices.get(offset) else {
                        continue;
                    };
                    let Some(triangle) = triangles.get(triangle_index) else {
                        continue;
                    };
                    let Some(hit) = intersect_triangle(
                        triangle,
                        triangle_index,
                        origin,
                        direction,
                        t_min,
                        current_max,
                        two_sided,
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
                let max_distance = nearest.map_or(t_max, |hit: BvhHit| hit.distance.min(t_max));
                let left_interval = self.nodes.get(left).and_then(|child| {
                    child
                        .bounds
                        .intersect_interval(origin, direction, t_min, max_distance)
                });
                let right_interval = self.nodes.get(right).and_then(|child| {
                    child
                        .bounds
                        .intersect_interval(origin, direction, t_min, max_distance)
                });

                match (left_interval, right_interval) {
                    (Some(left_interval), Some(right_interval)) => {
                        if left_interval.0 <= right_interval.0 {
                            self.visit(
                                left, triangles, origin, direction, t_min, t_max, two_sided,
                                nearest,
                            );
                            self.visit(
                                right, triangles, origin, direction, t_min, t_max, two_sided,
                                nearest,
                            );
                        } else {
                            self.visit(
                                right, triangles, origin, direction, t_min, t_max, two_sided,
                                nearest,
                            );
                            self.visit(
                                left, triangles, origin, direction, t_min, t_max, two_sided,
                                nearest,
                            );
                        }
                    }
                    (Some(_), None) => {
                        self.visit(
                            left, triangles, origin, direction, t_min, t_max, two_sided, nearest,
                        );
                    }
                    (None, Some(_)) => {
                        self.visit(
                            right, triangles, origin, direction, t_min, t_max, two_sided, nearest,
                        );
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
    triangles: &[Triangle],
    triangle_indices: &mut [usize],
    start: usize,
    nodes: &mut Vec<Node>,
) -> Result<usize> {
    let bounds = bounds_for_indices(triangles, triangle_indices)?;
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
        let left_center = Aabb::for_triangle(&triangles[*left]).center(axis);
        let right_center = Aabb::for_triangle(&triangles[*right]).center(axis);
        left_center
            .total_cmp(&right_center)
            .then_with(|| left.cmp(right))
    });

    let middle = triangle_indices.len() / 2;
    let right_start = start
        .checked_add(middle)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "BVH split index overflow"))?;
    let (left_indices, right_indices) = triangle_indices.split_at_mut(middle);
    let left = build_node(triangles, left_indices, start, nodes)?;
    let right = build_node(triangles, right_indices, right_start, nodes)?;
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

fn bounds_for_indices(triangles: &[Triangle], indices: &[usize]) -> Result<Aabb> {
    let Some(first_index) = indices.first().copied() else {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "cannot bound an empty BVH node",
        ));
    };
    let Some(first_triangle) = triangles.get(first_index) else {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "BVH triangle index is out of range",
        ));
    };
    let mut bounds = Aabb::for_triangle(first_triangle);
    for &index in &indices[1..] {
        let Some(triangle) = triangles.get(index) else {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "BVH triangle index is out of range",
            ));
        };
        bounds.include(Aabb::for_triangle(triangle));
    }
    Ok(bounds)
}

impl Aabb {
    fn for_triangle(triangle: &Triangle) -> Self {
        let mut min = triangle.positions[0];
        let mut max = triangle.positions[0];
        for point in &triangle.positions[1..] {
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

    fn intersect_interval(
        self,
        origin: DVec3,
        direction: DVec3,
        t_min: f64,
        t_max: f64,
    ) -> Option<(f64, f64)> {
        let mut near = t_min;
        let mut far = t_max;
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
}

fn intersect_triangle(
    triangle: &Triangle,
    triangle_index: usize,
    origin: DVec3,
    direction: DVec3,
    t_min: f64,
    t_max: f64,
    two_sided: bool,
) -> Option<BvhHit> {
    let edge1 = triangle.positions[1] - triangle.positions[0];
    let edge2 = triangle.positions[2] - triangle.positions[0];
    let p_vector = direction.cross(edge2);
    let determinant = edge1.dot(p_vector);
    if !determinant.is_finite()
        || determinant.abs() <= f64::MIN_POSITIVE
        || (!two_sided && !triangle.double_sided && determinant <= 0.0)
    {
        return None;
    }

    let inverse_determinant = 1.0 / determinant;
    if !inverse_determinant.is_finite() {
        return None;
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
        triangle_index,
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

    use crate::render::raster::{AlphaMode, Triangle};

    use super::{Bvh, BvhHit};

    proptest! {
        #[test]
        fn closest_bvh_hit_matches_brute_force_moller_trumbore(
            triangle_data in prop::collection::vec(
                (-8.0_f64..8.0, -8.0_f64..8.0, -8.0_f64..8.0, 0.1_f64..2.0, any::<bool>()),
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
                .map(|(index, (x, y, z, size, double_sided))| {
                    let center = DVec3::new(*x, *y, *z);
                    Triangle {
                        positions: [
                            center + DVec3::new(-*size, -*size, 0.0),
                            center + DVec3::new(*size, -*size, 0.0),
                            center + DVec3::new(0.0, *size, 0.0),
                        ],
                        object_index: index as u32,
                        element_index: index as u32,
                        color: [1.0; 4],
                        alpha_mode: AlphaMode::Opaque,
                        alpha_threshold: 0.5,
                        double_sided: index == target_index || *double_sided,
                        material_index: 0,
                        uv: [[0.0; 2]; 3],
                        uv_available: false,
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
            let actual = bvh.closest_hit_mode(&triangles, origin, ray_direction, t_min, t_max, false);
            let expected = brute_force_hit(&triangles, origin, ray_direction, t_min, t_max, false);
            prop_assert!(expected.is_some(), "constructed ray should hit its target triangle");
            prop_assert_eq!(actual, expected);
            let actual_two_sided =
                bvh.closest_hit_two_sided(&triangles, origin, ray_direction, t_min, t_max);
            let expected_two_sided =
                brute_force_hit(&triangles, origin, ray_direction, t_min, t_max, true);
            prop_assert_eq!(actual_two_sided, expected_two_sided);
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
