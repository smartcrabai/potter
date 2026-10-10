// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/mesh_remesh_voxel.cc, SPDX-FileCopyrightText: 2019 Blender Authors, GPL-2.0-or-later.
//
// Portions derived from OpenVDB (https://github.com/AcademySoftwareFoundation/openvdb):
//   openvdb/openvdb/tools/MeshToVolume.h, openvdb/openvdb/tools/VolumeToMesh.h, SPDX-FileCopyrightText: Copyright Contributors to the OpenVDB Project, Apache-2.0.

//! Blender-compatible voxel remeshing using a float32 level set and uniform volume meshing.
//!
//! The conversion follows `OpenVDB`'s `meshToLevelSet` narrow-band distance construction and
//! `volumeToMesh`'s per-cell edge-intersection averaging for adaptivity zero.

use glam::DVec3;
use serde_json::json;

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
};

// Blender explicitly overrides OpenVDB's default level-set half width of 3.0 with 1.0.
const LEVEL_SET_HALF_WIDTH: f32 = 1.0;
const RENORMALIZE_OFFSET: f32 = 0.8;
const VOXEL_BUDGET: usize = 16_000_000;
const BVH_LEAF_SIZE: usize = 8;

#[derive(Clone, Copy)]
struct Triangle {
    points: [DVec3; 3],
    min: DVec3,
    max: DVec3,
    centroid: DVec3,
}

#[derive(Clone, Copy)]
struct BvhNode {
    min: DVec3,
    max: DVec3,
    left: usize,
    right: usize,
    start: usize,
    end: usize,
}

impl BvhNode {
    fn leaf(min: DVec3, max: DVec3, start: usize, end: usize) -> Self {
        Self {
            min,
            max,
            left: usize::MAX,
            right: usize::MAX,
            start,
            end,
        }
    }

    fn is_leaf(self) -> bool {
        self.left == usize::MAX
    }
}

struct TriangleBvh {
    triangles: Vec<Triangle>,
    order: Vec<usize>,
    nodes: Vec<BvhNode>,
}

impl TriangleBvh {
    fn new(triangles: Vec<Triangle>) -> Self {
        let mut result = Self {
            order: (0..triangles.len()).collect(),
            triangles,
            nodes: Vec::new(),
        };
        if !result.triangles.is_empty() {
            result.build(0, result.order.len());
        }
        result
    }

    fn build(&mut self, start: usize, end: usize) -> usize {
        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        for &index in &self.order[start..end] {
            min = min.min(self.triangles[index].min);
            max = max.max(self.triangles[index].max);
        }
        let node_index = self.nodes.len();
        self.nodes.push(BvhNode::leaf(min, max, start, end));
        if end - start > BVH_LEAF_SIZE {
            let extent = max - min;
            let axis = if extent.x >= extent.y && extent.x >= extent.z {
                0
            } else if extent.y >= extent.z {
                1
            } else {
                2
            };
            let middle = start + (end - start) / 2;
            let triangles = &self.triangles;
            self.order[start..end].select_nth_unstable_by(middle - start, |left, right| {
                triangles[*left].centroid[axis]
                    .total_cmp(&triangles[*right].centroid[axis])
                    .then_with(|| left.cmp(right))
            });
            let left = self.build(start, middle);
            let right = self.build(middle, end);
            self.nodes[node_index] = BvhNode {
                min,
                max,
                left,
                right,
                start,
                end,
            };
        }
        node_index
    }

    fn closest_primitive(&self, point: DVec3) -> (f64, usize) {
        let mut best = f64::INFINITY;
        let mut primitive = usize::MAX;
        if !self.nodes.is_empty() {
            self.query(0, point, &mut best, &mut primitive);
        }
        (best, primitive)
    }

    #[expect(
        clippy::float_cmp,
        reason = "OpenVDB-compatible triangle selection uses exact float32 distance ties"
    )]
    fn query(&self, node_index: usize, point: DVec3, best: &mut f64, primitive: &mut usize) {
        let node = self.nodes[node_index];
        if (bounds_distance_squared(node.min, node.max, point) as f32) > (*best as f32) {
            return;
        }
        if node.is_leaf() {
            for &triangle_index in &self.order[node.start..node.end] {
                let triangle = self.triangles[triangle_index];
                let mut points = triangle.points;
                points.swap(1, 2);
                let distance = (point - closest_point_on_triangle(point, points)).length_squared();
                let distance_f32 = distance as f32;
                let best_f32 = *best as f32;
                if distance_f32 < best_f32 {
                    *best = distance;
                    *primitive = triangle_index;
                } else if distance_f32 == best_f32 {
                    *best = (*best).min(distance);
                    *primitive = (*primitive).min(triangle_index);
                }
            }
            return;
        }
        let left = self.nodes[node.left];
        let right = self.nodes[node.right];
        if bounds_distance_squared(left.min, left.max, point)
            <= bounds_distance_squared(right.min, right.max, point)
        {
            self.query(node.left, point, best, primitive);
            self.query(node.right, point, best, primitive);
        } else {
            self.query(node.right, point, best, primitive);
            self.query(node.left, point, best, primitive);
        }
    }
}

/// Evaluate Blender's VOXEL remesh at zero adaptivity.
pub(super) fn remesh(mesh: &Mesh, voxel_size: f32, smooth_shade: bool) -> Result<Mesh> {
    let triangles = mesh_triangles(mesh, voxel_size)?;
    if triangles.is_empty() {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "voxel remesh requires a mesh with at least one triangle",
        ));
    }
    let bvh = TriangleBvh::new(triangles.clone());
    let (min_coord, dimensions) = grid_dimensions(&triangles, voxel_size)?;
    let count = dimensions[0]
        .checked_mul(dimensions[1])
        .and_then(|count| count.checked_mul(dimensions[2]))
        .filter(|count| *count <= VOXEL_BUDGET)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "voxel remesh grid exceeds the configured voxel budget",
            )
        })?;
    let mut distance = vec![0.0_f32; count];
    let mut squared_distance = vec![0.0_f32; count];
    let mut closest_primitive = vec![usize::MAX; count];
    let mut known_sign = vec![false; count];
    let band = voxel_size * LEVEL_SET_HALF_WIDTH;
    let narrow_sign_radius = f64::from(LEVEL_SET_HALF_WIDTH + 1.0);

    for x in 0..dimensions[0] {
        for y in 0..dimensions[1] {
            for z in 0..dimensions[2] {
                let offset = grid_index([x, y, z], dimensions);
                let coord = [
                    min_coord[0] + x as i32,
                    min_coord[1] + y as i32,
                    min_coord[2] + z as i32,
                ];
                let point = DVec3::new(
                    f64::from(coord[0]),
                    f64::from(coord[1]),
                    f64::from(coord[2]),
                );
                let (squared, primitive) = bvh.closest_primitive(point);
                let squared = squared as f32;
                squared_distance[offset] = squared;
                closest_primitive[offset] = primitive;
                let index_distance = squared.sqrt();
                let world_distance = index_distance * voxel_size;
                let inside = if f64::from(index_distance) <= narrow_sign_radius {
                    known_sign[offset] = true;
                    point_inside_mesh(point, &bvh.triangles)
                } else {
                    false
                };
                distance[offset] = if inside {
                    -world_distance
                } else {
                    world_distance
                };
            }
        }
    }

    let mut active = initial_voxelization_mask(&bvh.triangles, min_coord, dimensions);
    let mut inside = fill_interior_signs(&mut distance, &known_sign, &active, dimensions, band);
    let mut signed_squared = squared_distance
        .iter()
        .zip(&inside)
        .map(|(squared, inside)| {
            if *inside || *squared <= 0.75_f32 {
                *squared
            } else {
                -*squared
            }
        })
        .collect::<Vec<_>>();
    repair_intersecting_voxels(
        &mut signed_squared,
        &mut active,
        &closest_primitive,
        &bvh.triangles,
        dimensions,
        min_coord,
    );
    for index in 0..count {
        if active[index] {
            inside[index] = signed_squared[index] >= 0.0;
            let magnitude = signed_squared[index].abs().sqrt() * voxel_size;
            distance[index] = if inside[index] { -magnitude } else { magnitude };
        } else {
            distance[index] = if inside[index] { -band } else { band };
        }
    }
    renormalize_level_set(&mut distance, &active, dimensions, voxel_size, band);
    volume_to_mesh(&distance, dimensions, min_coord, voxel_size, smooth_shade)
}

fn mesh_triangles(mesh: &Mesh, voxel_size: f32) -> Result<Vec<Triangle>> {
    let world_vertices = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.as_vec3())
        .collect::<Vec<_>>();
    let vertices = mesh
        .vertices
        .iter()
        .map(|vertex| {
            let co = vertex.co.as_vec3().to_array();
            DVec3::new(
                f64::from((f64::from(co[0]) / f64::from(voxel_size)) as f32),
                f64::from((f64::from(co[1]) / f64::from(voxel_size)) as f32),
                f64::from((f64::from(co[2]) / f64::from(voxel_size)) as f32),
            )
        })
        .collect::<Vec<_>>();
    let vertex_index = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<std::collections::HashMap<_, _>>();
    let fallback = mesh.triangulate().map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            format!("voxel remesh could not triangulate the input mesh: {error}"),
            json!({"mesh_error": error.to_string()}),
        )
    })?;
    let mut fallback_offset = 0;
    let mut triangles = Vec::new();
    for face in &mesh.faces {
        let triangle_count = face.vertices.len().saturating_sub(2);
        if face.vertices.len() == 3 {
            let points = face
                .vertices
                .iter()
                .map(|id| vertex_index.get(id).map(|index| vertices[*index]))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid_mesh("face references a missing vertex"))?;
            triangles.push(make_triangle([points[0], points[1], points[2]]));
        } else if face.vertices.len() == 4 {
            let points = face
                .vertices
                .iter()
                .map(|id| vertex_index.get(id).map(|index| vertices[*index]))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid_mesh("face references a missing vertex"))?;
            let world_points = face
                .vertices
                .iter()
                .map(|id| vertex_index.get(id).map(|index| world_vertices[*index]))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid_mesh("face references a missing vertex"))?;
            let p = [points[0], points[1], points[2], points[3]];
            let world = [
                world_points[0],
                world_points[1],
                world_points[2],
                world_points[3],
            ];
            let first_cross = (world[1] - world[0]).cross(world[2] - world[0]);
            let second_cross = (world[3] - world[0]).cross(world[2] - world[0]);
            if first_cross.dot(second_cross) > 0.0 {
                triangles.push(make_triangle([p[0], p[1], p[3]]));
                triangles.push(make_triangle([p[1], p[2], p[3]]));
            } else {
                triangles.push(make_triangle([p[0], p[1], p[2]]));
                triangles.push(make_triangle([p[0], p[2], p[3]]));
            }
        } else if face.vertices.len() > 4 {
            for triangle in &fallback[fallback_offset..fallback_offset + triangle_count] {
                let points = triangle
                    .iter()
                    .map(|id| vertex_index.get(id).map(|index| vertices[*index]))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| invalid_mesh("triangulation references a missing vertex"))?;
                triangles.push(make_triangle([points[0], points[1], points[2]]));
            }
        }
        fallback_offset += triangle_count;
    }
    Ok(triangles)
}

fn make_triangle(points: [DVec3; 3]) -> Triangle {
    let min = points[0].min(points[1]).min(points[2]);
    let max = points[0].max(points[1]).max(points[2]);
    Triangle {
        points,
        min,
        max,
        centroid: (points[0] + points[1] + points[2]) / 3.0,
    }
}

fn grid_dimensions(triangles: &[Triangle], voxel_size: f32) -> Result<([i32; 3], [usize; 3])> {
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    for triangle in triangles {
        min = min.min(triangle.min);
        max = max.max(triangle.max);
    }
    let pad = i32::try_from(LEVEL_SET_HALF_WIDTH.ceil() as i64 + 4)
        .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "voxel grid bounds overflow"))?;
    let lower = min
        .to_array()
        .map(|value| value.floor() as i64 - i64::from(pad));
    let upper = max
        .to_array()
        .map(|value| value.ceil() as i64 + i64::from(pad));
    if lower
        .iter()
        .chain(upper.iter())
        .any(|coordinate| *coordinate < i64::from(i32::MIN) || *coordinate > i64::from(i32::MAX))
    {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "voxel remesh grid coordinates exceed integer limits",
        ));
    }
    let lower = lower.map(|value| value as i32);
    let upper = upper.map(|value| value as i32);
    let dimensions = std::array::from_fn(|axis| {
        usize::try_from(i64::from(upper[axis]) - i64::from(lower[axis]) + 1).unwrap_or(usize::MAX)
    });
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "voxel size must be finite and positive",
        ));
    }
    Ok((lower, dimensions))
}

fn point_inside_mesh(point: DVec3, triangles: &[Triangle]) -> bool {
    let mut winding = 0.0;
    for triangle in triangles {
        let a = triangle.points[0] - point;
        let b = triangle.points[1] - point;
        let c = triangle.points[2] - point;
        let la = a.length();
        let lb = b.length();
        let lc = c.length();
        let numerator = a.dot(b.cross(c));
        let denominator = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
        winding += 2.0 * numerator.atan2(denominator);
    }
    winding.abs() > std::f64::consts::TAU
}

fn initial_voxelization_mask(
    triangles: &[Triangle],
    min_coord: [i32; 3],
    dimensions: [usize; 3],
) -> Vec<bool> {
    let count = dimensions[0] * dimensions[1] * dimensions[2];
    let mut active = vec![false; count];
    let mut visited = vec![0_usize; count];
    let mut queue = std::collections::VecDeque::new();
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        let tag = triangle_index + 1;
        let seed = [
            triangle.points[0].x.floor() as i32 - min_coord[0],
            triangle.points[0].y.floor() as i32 - min_coord[1],
            triangle.points[0].z.floor() as i32 - min_coord[2],
        ];
        if !seed
            .iter()
            .enumerate()
            .all(|(axis, value)| *value >= 0 && *value < dimensions[axis] as i32)
        {
            continue;
        }
        queue.clear();
        let seed = [seed[0] as usize, seed[1] as usize, seed[2] as usize];
        let seed_index = grid_index(seed, dimensions);
        visited[seed_index] = tag;
        active[seed_index] = true;
        queue.push_back(seed);
        while let Some(coord) = queue.pop_back() {
            for dx in -1_i32..=1 {
                for dy in -1_i32..=1 {
                    for dz in -1_i32..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let neighbor = [
                            coord[0] as i32 + dx,
                            coord[1] as i32 + dy,
                            coord[2] as i32 + dz,
                        ];
                        if !neighbor
                            .iter()
                            .enumerate()
                            .all(|(axis, value)| *value >= 0 && *value < dimensions[axis] as i32)
                        {
                            continue;
                        }
                        let neighbor = [
                            neighbor[0] as usize,
                            neighbor[1] as usize,
                            neighbor[2] as usize,
                        ];
                        let neighbor_index = grid_index(neighbor, dimensions);
                        if visited[neighbor_index] == tag {
                            continue;
                        }
                        visited[neighbor_index] = tag;
                        active[neighbor_index] = true;
                        let point = DVec3::new(
                            f64::from(min_coord[0] + neighbor[0] as i32),
                            f64::from(min_coord[1] + neighbor[1] as i32),
                            f64::from(min_coord[2] + neighbor[2] as i32),
                        );
                        let mut ordered_points = triangle.points;
                        ordered_points.swap(1, 2);
                        let squared = (point - closest_point_on_triangle(point, ordered_points))
                            .length_squared() as f32;
                        if squared <= 0.75_f32 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }
    }
    active
}

fn fill_interior_signs(
    distance: &mut [f32],
    known_sign: &[bool],
    active: &[bool],
    dimensions: [usize; 3],
    band: f32,
) -> Vec<bool> {
    let count = distance.len();
    let mut exterior = vec![false; count];
    let mut inside = vec![false; count];
    let mut queue = std::collections::VecDeque::new();
    for x in 0..dimensions[0] {
        for y in 0..dimensions[1] {
            for z in 0..dimensions[2] {
                if x != 0
                    && y != 0
                    && z != 0
                    && x + 1 != dimensions[0]
                    && y + 1 != dimensions[1]
                    && z + 1 != dimensions[2]
                {
                    continue;
                }
                let offset = grid_index([x, y, z], dimensions);
                if !known_sign[offset] && !exterior[offset] {
                    exterior[offset] = true;
                    queue.push_back([x, y, z]);
                }
            }
        }
    }
    while let Some([x, y, z]) = queue.pop_front() {
        for neighbor in neighbors([x, y, z], dimensions) {
            let offset = grid_index(neighbor, dimensions);
            if !known_sign[offset] && !exterior[offset] {
                exterior[offset] = true;
                queue.push_back(neighbor);
            }
        }
    }
    for offset in 0..count {
        inside[offset] = if known_sign[offset] {
            distance[offset] < 0.0
        } else {
            !exterior[offset]
        };
        if active[offset] {
            distance[offset] = if inside[offset] {
                -distance[offset].abs()
            } else {
                distance[offset].abs()
            };
        } else {
            distance[offset] = if inside[offset] { -band } else { band };
        }
    }
    inside
}

fn repair_intersecting_voxels(
    signed_squared: &mut [f32],
    active: &mut [bool],
    closest_primitive: &[usize],
    triangles: &[Triangle],
    dimensions: [usize; 3],
    min_coord: [i32; 3],
) {
    for index in 0..signed_squared.len() {
        if !active[index] || signed_squared[index] < 0.0 || signed_squared[index] > 0.75 {
            continue;
        }
        let coord = [
            index / (dimensions[1] * dimensions[2]),
            (index / dimensions[2]) % dimensions[1],
            index % dimensions[2],
        ];
        let current = DVec3::new(
            f64::from(min_coord[0] + coord[0] as i32),
            f64::from(min_coord[1] + coord[1] as i32),
            f64::from(min_coord[2] + coord[2] as i32),
        );
        let mut flip_sign = false;
        for dx in -1_i32..=1 {
            for dy in -1_i32..=1 {
                for dz in -1_i32..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let neighbor = [
                        coord[0] as i32 + dx,
                        coord[1] as i32 + dy,
                        coord[2] as i32 + dz,
                    ];
                    if !neighbor
                        .iter()
                        .enumerate()
                        .all(|(axis, value)| *value >= 0 && *value < dimensions[axis] as i32)
                    {
                        continue;
                    }
                    let neighbor = [
                        neighbor[0] as usize,
                        neighbor[1] as usize,
                        neighbor[2] as usize,
                    ];
                    let neighbor_index = grid_index(neighbor, dimensions);
                    if !active[neighbor_index] || signed_squared[neighbor_index] >= -0.75 {
                        continue;
                    }
                    let primitive = closest_primitive[neighbor_index];
                    if primitive >= triangles.len() {
                        continue;
                    }
                    let neighbor_point = DVec3::new(
                        f64::from(min_coord[0] + neighbor[0] as i32),
                        f64::from(min_coord[1] + neighbor[1] as i32),
                        f64::from(min_coord[2] + neighbor[2] as i32),
                    );
                    let mut points = triangles[primitive].points;
                    points.swap(1, 2);
                    let closest = closest_point_on_triangle(neighbor_point, points);
                    let from_neighbor = (neighbor_point - closest).normalize_or_zero();
                    let from_current = (current - closest).normalize_or_zero();
                    if from_neighbor.dot(from_current) > 0.0 {
                        flip_sign = true;
                        break;
                    }
                }
                if flip_sign {
                    break;
                }
            }
            if flip_sign {
                break;
            }
        }
        if flip_sign {
            signed_squared[index] = -signed_squared[index];
        }
    }

    for index in 0..signed_squared.len() {
        if !active[index] || signed_squared[index] < 0.0 || signed_squared[index] > 0.75 {
            continue;
        }
        let coord = [
            index / (dimensions[1] * dimensions[2]),
            (index / dimensions[2]) % dimensions[1],
            index % dimensions[2],
        ];
        let has_negative_neighbor = neighbors(coord, dimensions).any(|neighbor| {
            let neighbor_index = grid_index(neighbor, dimensions);
            active[neighbor_index] && signed_squared[neighbor_index] < 0.0
        });
        if !has_negative_neighbor {
            signed_squared[index] = 0.75 + 1.0e-7_f32;
        }
    }
    for index in 0..signed_squared.len() {
        if !active[index] || signed_squared[index] <= 0.75 {
            continue;
        }
        let coord = [
            index / (dimensions[1] * dimensions[2]),
            (index / dimensions[2]) % dimensions[1],
            index % dimensions[2],
        ];
        let has_boundary_neighbor = neighbors(coord, dimensions).any(|neighbor| {
            let neighbor_index = grid_index(neighbor, dimensions);
            active[neighbor_index] && signed_squared[neighbor_index] <= 0.75
        });
        if !has_boundary_neighbor {
            active[index] = false;
        }
    }
}
fn renormalize_level_set(
    distance: &mut [f32],
    active: &[bool],
    dimensions: [usize; 3],
    voxel_size: f32,
    band: f32,
) {
    let offset = (f64::from(RENORMALIZE_OFFSET) * f64::from(voxel_size)) as f32;
    let mut shifted = distance.to_vec();
    for (value, is_active) in shifted.iter_mut().zip(active) {
        if *is_active {
            *value -= offset;
        }
    }
    let mut corrected = shifted.clone();
    let inv_voxel_size = 1.0_f32 / voxel_size;
    for x in 0..dimensions[0] {
        for y in 0..dimensions[1] {
            for z in 0..dimensions[2] {
                let index = grid_index([x, y, z], dimensions);
                if !active[index] {
                    continue;
                }
                let phi = shifted[index];
                let down = [
                    phi - shifted[grid_index([x.saturating_sub(1), y, z], dimensions)],
                    phi - shifted[grid_index([x, y.saturating_sub(1), z], dimensions)],
                    phi - shifted[grid_index([x, y, z.saturating_sub(1)], dimensions)],
                ];
                let up = [
                    shifted[grid_index([(x + 1).min(dimensions[0] - 1), y, z], dimensions)] - phi,
                    shifted[grid_index([x, (y + 1).min(dimensions[1] - 1), z], dimensions)] - phi,
                    shifted[grid_index([x, y, (z + 1).min(dimensions[2] - 1)], dimensions)] - phi,
                ];
                let norm_squared = godunov_norm_squared(phi > 0.0, down, up);
                let norm = norm_squared.sqrt();
                let difference = norm * inv_voxel_size - 1.0;
                let sign = phi / (phi * phi + norm_squared).sqrt();
                let update = phi - voxel_size * sign * difference;
                corrected[index] = shifted[index].min(update) + (offset - 1.0e-7_f32);
            }
        }
    }
    for index in 0..distance.len() {
        let value = if active[index] {
            corrected[index]
        } else {
            distance[index]
        };
        distance[index] = value.clamp(-band, band);
    }
}

fn godunov_norm_squared(outside: bool, down: [f32; 3], up: [f32; 3]) -> f32 {
    let mut result = 0.0;
    for axis in 0..3 {
        let (backward, forward) = if outside {
            (down[axis].max(0.0), up[axis].min(0.0))
        } else {
            (down[axis].min(0.0), up[axis].max(0.0))
        };
        result += backward.powi(2).max(forward.powi(2));
    }
    result
}

#[derive(Clone, Copy, Default)]
struct Cell {
    groups: [u8; 12],
    point_offset: usize,
    point_count: usize,
}

fn volume_to_mesh(
    values: &[f32],
    dimensions: [usize; 3],
    min_coord: [i32; 3],
    voxel_size: f32,
    smooth_shade: bool,
) -> Result<Mesh> {
    let cell_dimensions = [dimensions[0] - 1, dimensions[1] - 1, dimensions[2] - 1];
    let cell_count = cell_dimensions[0]
        .checked_mul(cell_dimensions[1])
        .and_then(|count| count.checked_mul(cell_dimensions[2]))
        .filter(|count| *count <= VOXEL_BUDGET)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "voxel remesh cell grid is too large",
            )
        })?;
    let mut cells = vec![Cell::default(); cell_count];
    let mut positions = Vec::<DVec3>::new();

    for x in 0..cell_dimensions[0] {
        for y in 0..cell_dimensions[1] {
            for z in 0..cell_dimensions[2] {
                let coord = [x, y, z];
                let cell_values = cell_corner_values(values, dimensions, coord);
                let signs = corrected_cell_signs(values, dimensions, coord);
                if signs == 0 || signs == u8::MAX {
                    continue;
                }
                let (groups, count) = edge_groups(signs);
                let point_offset = positions.len();
                for group in 1..=count {
                    let point = compute_cell_point(&cell_values, &groups, group);
                    let index_position = DVec3::new(
                        f64::from(min_coord[0] + x as i32) + point.x,
                        f64::from(min_coord[1] + y as i32) + point.y,
                        f64::from(min_coord[2] + z as i32) + point.z,
                    );
                    let world = index_position * f64::from(voxel_size);
                    positions.push(DVec3::new(
                        f64::from(world.x as f32),
                        f64::from(world.y as f32),
                        f64::from(world.z as f32),
                    ));
                }
                cells[cell_index(coord, cell_dimensions)] = Cell {
                    groups,
                    point_offset,
                    point_count: usize::from(count),
                };
            }
        }
    }

    let mut polygons = Vec::<Vec<usize>>::new();
    for x in 0..cell_dimensions[0] {
        for y in 0..cell_dimensions[1] {
            for z in 0..cell_dimensions[2] {
                let coord = [x, y, z];
                let cell = cells[cell_index(coord, cell_dimensions)];
                if cell.point_count == 0 {
                    continue;
                }
                let signs = corrected_cell_signs(values, dimensions, coord);
                if (signs & 1 != 0) != (signs & 2 != 0) {
                    add_quad(
                        &mut polygons,
                        &cells,
                        cell_dimensions,
                        coord,
                        [(0, 0, 0, 1), (0, -1, 0, 5), (0, -1, -1, 7), (0, 0, -1, 3)],
                    );
                }
                if (signs & 1 != 0) != (signs & 16 != 0) {
                    add_quad(
                        &mut polygons,
                        &cells,
                        cell_dimensions,
                        coord,
                        [
                            (0, 0, 0, 9),
                            (0, 0, -1, 12),
                            (-1, 0, -1, 11),
                            (-1, 0, 0, 10),
                        ],
                    );
                }
                if (signs & 1 != 0) != (signs & 8 != 0) {
                    add_quad(
                        &mut polygons,
                        &cells,
                        cell_dimensions,
                        coord,
                        [(0, 0, 0, 4), (0, -1, 0, 8), (-1, -1, 0, 6), (-1, 0, 0, 2)],
                    );
                }
            }
        }
    }
    if polygons.is_empty() {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "voxel remesh did not produce a surface",
        ));
    }
    let mut output = Mesh::from_positions_and_faces(positions, polygons).map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            format!("voxel remesh produced invalid geometry: {error}"),
            json!({"mesh_error": error.to_string()}),
        )
    })?;
    if smooth_shade {
        let values = output
            .faces
            .iter()
            .map(|face| (format!("f{}", face.id), json!(true)))
            .collect::<serde_json::Map<_, _>>();
        output.attributes.insert(
            "shade_smooth".to_owned(),
            json!({"domain":"faces","type":"bool","values":values}),
        );
    }
    Ok(output)
}

fn add_quad(
    polygons: &mut Vec<Vec<usize>>,
    cells: &[Cell],
    dimensions: [usize; 3],
    origin: [usize; 3],
    adjacent: [(i32, i32, i32, usize); 4],
) {
    let mut polygon = Vec::with_capacity(4);
    for (dx, dy, dz, group_edge) in adjacent {
        let neighbor = [
            origin[0] as i32 + dx,
            origin[1] as i32 + dy,
            origin[2] as i32 + dz,
        ];
        if neighbor
            .iter()
            .enumerate()
            .any(|(axis, value)| *value < 0 || *value >= dimensions[axis] as i32)
        {
            return;
        }
        let cell = cells[cell_index(
            [
                neighbor[0] as usize,
                neighbor[1] as usize,
                neighbor[2] as usize,
            ],
            dimensions,
        )];
        if cell.point_count == 0 {
            return;
        }
        let group = cell.groups[group_edge - 1];
        if group == 0 || usize::from(group) > cell.point_count {
            return;
        }
        polygon.push(cell.point_offset + usize::from(group) - 1);
    }
    polygons.push(polygon);
}

fn cell_corner_values(values: &[f32], dimensions: [usize; 3], coord: [usize; 3]) -> [f64; 8] {
    let points = [
        [coord[0], coord[1], coord[2]],
        [coord[0] + 1, coord[1], coord[2]],
        [coord[0] + 1, coord[1], coord[2] + 1],
        [coord[0], coord[1], coord[2] + 1],
        [coord[0], coord[1] + 1, coord[2]],
        [coord[0] + 1, coord[1] + 1, coord[2]],
        [coord[0] + 1, coord[1] + 1, coord[2] + 1],
        [coord[0], coord[1] + 1, coord[2] + 1],
    ];
    std::array::from_fn(|index| f64::from(values[grid_index(points[index], dimensions)]))
}

fn cell_signs(values: &[f64; 8]) -> u8 {
    values
        .iter()
        .enumerate()
        .fold(0_u8, |signs, (index, value)| {
            signs | (u8::from(*value < 0.0) << index)
        })
}

fn corrected_cell_signs(values: &[f32], dimensions: [usize; 3], coord: [usize; 3]) -> u8 {
    use super::voxel_edge_groups::AMBIGUOUS_FACE;

    let cell_dimensions = [dimensions[0] - 1, dimensions[1] - 1, dimensions[2] - 1];
    let mut signs = cell_signs(&cell_corner_values(values, dimensions, coord));
    let (neighbor, expected_face) = match AMBIGUOUS_FACE[usize::from(signs)] {
        1 if coord[2] > 0 => ([coord[0], coord[1], coord[2] - 1], 3),
        2 if coord[0] + 1 < cell_dimensions[0] => ([coord[0] + 1, coord[1], coord[2]], 4),
        3 if coord[2] + 1 < cell_dimensions[2] => ([coord[0], coord[1], coord[2] + 1], 1),
        4 if coord[0] > 0 => ([coord[0] - 1, coord[1], coord[2]], 2),
        5 if coord[1] > 0 => ([coord[0], coord[1] - 1, coord[2]], 6),
        6 if coord[1] + 1 < cell_dimensions[1] => ([coord[0], coord[1] + 1, coord[2]], 5),
        _ => return signs,
    };
    let neighbor_signs = cell_signs(&cell_corner_values(values, dimensions, neighbor));
    if AMBIGUOUS_FACE[usize::from(neighbor_signs)] == expected_face {
        signs = !signs;
    }
    signs
}

fn edge_groups(signs: u8) -> ([u8; 12], u8) {
    let group_row = super::voxel_edge_groups::EDGE_GROUPS[usize::from(signs)];
    (
        std::array::from_fn(|edge| group_row[edge + 1]),
        group_row[0],
    )
}

fn compute_cell_point(values: &[f64; 8], groups: &[u8; 12], group: u8) -> DVec3 {
    const EDGES: [(usize, usize, [f64; 3], [f64; 3]); 12] = [
        (0, 1, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        (1, 2, [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]),
        (3, 2, [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
        (0, 3, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        (4, 5, [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]),
        (5, 6, [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]),
        (7, 6, [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        (4, 7, [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]),
        (0, 4, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        (1, 5, [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]),
        (2, 6, [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
        (3, 7, [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]),
    ];
    let mut point = DVec3::ZERO;
    let mut samples = 0.0;
    for (edge, (a, b, start, end)) in EDGES.into_iter().enumerate() {
        if groups[edge] != group {
            continue;
        }
        let t = (0.0 - values[a]) / (values[b] - values[a]);
        point += DVec3::from_array(start) + (DVec3::from_array(end) - DVec3::from_array(start)) * t;
        samples += 1.0;
    }
    if samples > 1.0 {
        point /= samples;
    }
    point
}

fn closest_point_on_triangle(point: DVec3, triangle: [DVec3; 3]) -> DVec3 {
    let [a, b, c] = triangle;
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denominator = 1.0 / (va + vb + vc);
    let v = vb * denominator;
    let w = vc * denominator;
    a + ab * v + ac * w
}

fn bounds_distance_squared(min: DVec3, max: DVec3, point: DVec3) -> f64 {
    let delta = (min - point).max(DVec3::ZERO) + (point - max).max(DVec3::ZERO);
    delta.length_squared()
}

fn grid_index(coord: [usize; 3], dimensions: [usize; 3]) -> usize {
    (coord[0] * dimensions[1] + coord[1]) * dimensions[2] + coord[2]
}

fn cell_index(coord: [usize; 3], dimensions: [usize; 3]) -> usize {
    grid_index(coord, dimensions)
}

fn neighbors(coord: [usize; 3], dimensions: [usize; 3]) -> impl Iterator<Item = [usize; 3]> {
    let mut adjacent = [[0; 3]; 6];
    let mut count = 0;
    for axis in 0..3 {
        if coord[axis] > 0 {
            let mut neighbor = coord;
            neighbor[axis] -= 1;
            adjacent[count] = neighbor;
            count += 1;
        }
        if coord[axis] + 1 < dimensions[axis] {
            let mut neighbor = coord;
            neighbor[axis] += 1;
            adjacent[count] = neighbor;
            count += 1;
        }
    }
    adjacent.into_iter().take(count)
}

fn invalid_mesh(message: &str) -> PotError {
    PotError::new(ErrorCode::InvalidArgument, message)
}
