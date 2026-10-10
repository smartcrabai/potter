//! Signed-distance voxel remeshing for closed polygon meshes.

use std::collections::{HashMap, HashSet};

use glam::DVec3;

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Edge, Face, Mesh, MeshError},
};

use super::edge_key;

const MAX_GRID_SAMPLES: usize = 1_000_000;
const MAX_GRID_CELLS: usize = 250_000;
const MAX_GRID_AXIS_F64: f64 = 512.0;
const MAX_TRIANGLES: usize = 100_000;
const MAX_DISTANCE_TESTS: usize = 100_000_000;
const MAX_OUTPUT_FACES: usize = 500_000;
const CUBE_EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 3],
    [3, 2],
    [2, 0],
    [4, 5],
    [5, 7],
    [7, 6],
    [6, 4],
    [0, 4],
    [1, 5],
    [3, 7],
    [2, 6],
];
const CUBE_FACES: [([usize; 4], [usize; 4]); 6] = [
    ([0, 1, 3, 2], [0, 1, 2, 3]),
    ([4, 5, 7, 6], [4, 5, 6, 7]),
    ([0, 1, 5, 4], [0, 9, 4, 8]),
    ([1, 3, 7, 5], [1, 10, 5, 9]),
    ([2, 3, 7, 6], [2, 10, 6, 11]),
    ([0, 2, 6, 4], [3, 11, 7, 8]),
];

/// Reconstruct a closed mesh from its signed-distance field at a regular voxel grid.
///
/// Marching-cubes contours use all twelve cube edges; a shared-face scalar decider
/// resolves ambiguous cases consistently. Crossing vertices are cached by lattice edge.
///
/// # Errors
///
/// Returns an error for invalid voxel sizes, invalid or open input meshes, exhausted
/// geometry resources, or failures while constructing the output mesh.
pub fn remesh(mesh: &Mesh, voxel_size: f64) -> Result<Mesh> {
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        return Err(PotError::invalid_argument(
            "voxel_size must be finite and positive",
        ));
    }
    mesh.validate().map_err(|error| mesh_input_error(&error))?;
    let bounds = mesh.bounds().ok_or_else(|| {
        PotError::invalid_operation("voxel remesh requires a non-empty closed mesh")
    })?;
    if mesh.faces.is_empty() {
        return Err(PotError::invalid_operation(
            "voxel remesh requires a non-empty closed mesh",
        ));
    }
    let mut boundary_use = HashMap::new();
    let mut surface_vertices = HashSet::with_capacity(mesh.vertices.len());
    for face in &mesh.faces {
        for index in 0..face.vertices.len() {
            let first = face.vertices[index];
            let second = face.vertices[(index + 1) % face.vertices.len()];
            surface_vertices.insert(first);
            let key = if first < second {
                (first, second)
            } else {
                (second, first)
            };
            let count = boundary_use.entry(key).or_insert(0_u8);
            *count = count.saturating_add(1);
        }
    }
    if surface_vertices.len() != mesh.vertices.len()
        || boundary_use.values().any(|&count| count != 2)
        || mesh.edges.iter().any(|edge| {
            let key = if edge.vertices[0] < edge.vertices[1] {
                (edge.vertices[0], edge.vertices[1])
            } else {
                (edge.vertices[1], edge.vertices[0])
            };
            boundary_use.get(&key) != Some(&2)
        })
    {
        return Err(PotError::invalid_operation(
            "voxel remesh requires a closed two-manifold input mesh",
        ));
    }
    let triangle_count = mesh
        .faces
        .iter()
        .try_fold(0_usize, |total, face| {
            total.checked_add(face.vertices.len() - 2)
        })
        .ok_or_else(resource_error)?;
    if triangle_count == 0 || triangle_count > MAX_TRIANGLES {
        return Err(resource_error());
    }
    let triangles = mesh
        .triangulate()
        .map_err(|error| mesh_input_error(&error))?;
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut surface = Vec::with_capacity(triangles.len());
    for triangle in triangles {
        let points = [
            *positions.get(&triangle[0]).ok_or_else(|| {
                PotError::invalid_operation("input triangle references a missing vertex")
            })?,
            *positions.get(&triangle[1]).ok_or_else(|| {
                PotError::invalid_operation("input triangle references a missing vertex")
            })?,
            *positions.get(&triangle[2]).ok_or_else(|| {
                PotError::invalid_operation("input triangle references a missing vertex")
            })?,
        ];
        let normal = (points[1] - points[0]).cross(points[2] - points[0]);
        if !normal.is_finite() || normal.length_squared() <= f64::MIN_POSITIVE {
            return Err(PotError::invalid_operation(
                "voxel remesh input contains a degenerate triangle",
            ));
        }
        surface.push(points);
    }

    let grid_min = bounds.min - DVec3::splat(voxel_size);
    if !grid_min.is_finite() || (0..3).any(|axis| grid_min[axis] + voxel_size <= grid_min[axis]) {
        return Err(resource_error());
    }
    let nx = grid_axis(bounds.size().x, voxel_size)?;
    let ny = grid_axis(bounds.size().y, voxel_size)?;
    let nz = grid_axis(bounds.size().z, voxel_size)?;
    let grid_max = grid_point(grid_min, voxel_size, nx - 1, ny - 1, nz - 1);
    if !grid_max.is_finite() {
        return Err(resource_error());
    }
    let sample_count = nx
        .checked_mul(ny)
        .and_then(|count| count.checked_mul(nz))
        .ok_or_else(resource_error)?;
    let cell_count = (nx - 1)
        .checked_mul(ny - 1)
        .and_then(|count| count.checked_mul(nz - 1))
        .ok_or_else(resource_error)?;
    if sample_count > MAX_GRID_SAMPLES || cell_count > MAX_GRID_CELLS {
        return Err(resource_error());
    }
    let distance_tests = sample_count
        .checked_mul(surface.len())
        .ok_or_else(resource_error)?;
    if distance_tests > MAX_DISTANCE_TESTS {
        return Err(resource_error());
    }
    let mut distances = Vec::with_capacity(sample_count);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let point = grid_point(grid_min, voxel_size, x, y, z);
                let distance = signed_distance(point, &surface);
                if !distance.is_finite() {
                    return Err(resource_error());
                }
                distances.push(distance);
            }
        }
    }

    let mut output = Mesh::new();
    let mut output_edges = HashSet::new();
    let mut crossings: HashMap<(usize, usize), (u32, DVec3)> = HashMap::new();
    let iso_level = voxel_size * 1.0e-10;
    for z in 0..nz - 1 {
        for y in 0..ny - 1 {
            for x in 0..nx - 1 {
                let corners = cube_corners(x, y, z, nx, ny);
                let cell_origin = grid_point(grid_min, voxel_size, x, y, z);
                let values = corners.map(|node| distances[node] - iso_level);
                let mut inside_center = DVec3::ZERO;
                let mut outside_center = DVec3::ZERO;
                let mut inside_count = 0_u8;
                let mut outside_count = 0_u8;
                for corner in 0..8 {
                    let position =
                        grid_point_from_index(corners[corner], nx, ny, grid_min, voxel_size);
                    if values[corner] < 0.0 {
                        inside_center += position;
                        inside_count += 1;
                    } else {
                        outside_center += position;
                        outside_count += 1;
                    }
                }
                if inside_count == 0 || outside_count == 0 {
                    continue;
                }
                let fallback_outward = outside_center / f64::from(outside_count)
                    - inside_center / f64::from(inside_count);
                let mut edge_points = [None; 12];
                for (edge_index, [first, second]) in CUBE_EDGES.iter().copied().enumerate() {
                    if (values[first] < 0.0) == (values[second] < 0.0) {
                        continue;
                    }
                    let first_node = corners[first];
                    let second_node = corners[second];
                    let edge_key = if first_node < second_node {
                        (first_node, second_node)
                    } else {
                        (second_node, first_node)
                    };
                    let crossing = if let Some(&cached) = crossings.get(&edge_key) {
                        cached
                    } else {
                        let fraction =
                            (-values[first] / (values[second] - values[first])).clamp(0.0, 1.0);
                        let first_position =
                            grid_point_from_index(first_node, nx, ny, grid_min, voxel_size);
                        let second_position =
                            grid_point_from_index(second_node, nx, ny, grid_min, voxel_size);
                        let position = first_position.lerp(second_position, fraction);
                        let id = output
                            .insert_vertex(position)
                            .map_err(|error| mesh_output_error(&error))?;
                        crossings.insert(edge_key, (id, position));
                        (id, position)
                    };
                    edge_points[edge_index] = Some(crossing);
                }

                let mut neighbors = [[usize::MAX; 2]; 12];
                let mut degree = [0_u8; 12];
                for (face_corners, face_edges) in CUBE_FACES {
                    let mut crossed_edges = [usize::MAX; 4];
                    let mut crossed_count = 0;
                    for edge in face_edges {
                        if edge_points[edge].is_some() {
                            crossed_edges[crossed_count] = edge;
                            crossed_count += 1;
                        }
                    }
                    match crossed_count {
                        0 => {}
                        2 => add_neighbor(
                            &mut neighbors,
                            &mut degree,
                            crossed_edges[0],
                            crossed_edges[1],
                        )?,
                        4 => {
                            let center_inside = face_corners
                                .iter()
                                .map(|&corner| values[corner])
                                .sum::<f64>()
                                < 0.0;
                            for corner in 0..4 {
                                if (values[face_corners[corner]] < 0.0) != center_inside {
                                    let previous_edge = face_edges[(corner + 3) % 4];
                                    let next_edge = face_edges[corner];
                                    add_neighbor(
                                        &mut neighbors,
                                        &mut degree,
                                        previous_edge,
                                        next_edge,
                                    )?;
                                }
                            }
                        }
                        _ => {
                            return Err(PotError::new(
                                ErrorCode::InternalError,
                                "marching-cubes face has an invalid crossing count",
                            ));
                        }
                    }
                }

                let mut visited = [false; 12];
                for start in 0..12 {
                    if edge_points[start].is_none() || visited[start] {
                        continue;
                    }
                    let mut loop_edges = [usize::MAX; 12];
                    let mut loop_len = 0;
                    let mut previous = usize::MAX;
                    let mut current = start;
                    let mut closed = false;
                    for _ in 0..12 {
                        if visited[current] {
                            break;
                        }
                        visited[current] = true;
                        loop_edges[loop_len] = current;
                        loop_len += 1;
                        let [first_neighbor, second_neighbor] = neighbors[current];
                        if first_neighbor == usize::MAX || second_neighbor == usize::MAX {
                            return Err(PotError::new(
                                ErrorCode::InternalError,
                                "marching-cubes contour is not closed",
                            ));
                        }
                        let next = if first_neighbor == previous {
                            second_neighbor
                        } else {
                            first_neighbor
                        };
                        if next == start {
                            closed = true;
                            break;
                        }
                        previous = current;
                        current = next;
                    }
                    if !closed {
                        return Err(PotError::new(
                            ErrorCode::InternalError,
                            "marching-cubes contour traversal did not close",
                        ));
                    }
                    if loop_len < 3 {
                        continue;
                    }
                    let mut loop_points = [(0_u32, DVec3::ZERO); 12];
                    for index in 0..loop_len {
                        loop_points[index] = edge_points[loop_edges[index]].ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "marching-cubes contour vertex is missing",
                            )
                        })?;
                    }
                    for index in 1..loop_len - 1 {
                        let triangle = [loop_points[0], loop_points[index], loop_points[index + 1]];
                        let center = (triangle[0].1 + triangle[1].1 + triangle[2].1) / 3.0;
                        let local = (center - cell_origin) / voxel_size;
                        let gradient = trilinear_gradient(values, local, voxel_size);
                        let outward = if gradient.length_squared() > f64::MIN_POSITIVE {
                            gradient
                        } else {
                            fallback_outward
                        };
                        add_oriented_triangle(
                            &mut output,
                            &mut output_edges,
                            triangle[0],
                            triangle[1],
                            triangle[2],
                            outward,
                        )?;
                    }
                }
            }
        }
    }
    if output.faces.is_empty() {
        return Err(PotError::invalid_operation(
            "voxel remesh produced no surface; use a larger voxel_size",
        ));
    }
    Ok(output)
}

fn add_neighbor(
    neighbors: &mut [[usize; 2]; 12],
    degree: &mut [u8; 12],
    first: usize,
    second: usize,
) -> Result<()> {
    if first == second || degree[first] >= 2 || degree[second] >= 2 {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "marching-cubes contour edge has invalid incidence",
        ));
    }
    neighbors[first][usize::from(degree[first])] = second;
    neighbors[second][usize::from(degree[second])] = first;
    degree[first] += 1;
    degree[second] += 1;
    Ok(())
}
fn mesh_input_error(error: &MeshError) -> PotError {
    PotError::new(
        ErrorCode::InvalidOperation,
        format!("invalid voxel remesh input: {error}"),
    )
}

fn mesh_output_error(error: &MeshError) -> PotError {
    PotError::new(
        ErrorCode::LimitExceeded,
        format!("voxel remesh output exceeds mesh limits: {error}"),
    )
}

fn resource_error() -> PotError {
    PotError::new(
        ErrorCode::LimitExceeded,
        "voxel remesh grid or geometry exceeds resource limits",
    )
}

fn grid_axis(extent: f64, voxel_size: f64) -> Result<usize> {
    let axis = (extent / voxel_size).ceil() + 3.0;
    if !axis.is_finite() || axis < 3.0 || axis > MAX_GRID_AXIS_F64 {
        return Err(resource_error());
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "grid axis is checked positive and against a small finite maximum"
    )]
    Ok(axis as usize)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "grid indices are explicitly bounded to at most 512"
)]
fn grid_point(origin: DVec3, voxel_size: f64, x: usize, y: usize, z: usize) -> DVec3 {
    origin + voxel_size * DVec3::new(x as f64, y as f64, z as f64)
}

fn grid_point_from_index(
    index: usize,
    nx: usize,
    ny: usize,
    origin: DVec3,
    voxel_size: f64,
) -> DVec3 {
    let x = index % nx;
    let y = (index / nx) % ny;
    let z = index / (nx * ny);
    grid_point(origin, voxel_size, x, y, z)
}

fn cube_corners(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> [usize; 8] {
    let base = x + nx * (y + ny * z);
    let stride_y = nx;
    let stride_z = nx * ny;
    [
        base,
        base + 1,
        base + stride_y,
        base + stride_y + 1,
        base + stride_z,
        base + stride_z + 1,
        base + stride_z + stride_y,
        base + stride_z + stride_y + 1,
    ]
}
fn trilinear_gradient(values: [f64; 8], point: DVec3, voxel_size: f64) -> DVec3 {
    let x = point.x.clamp(0.0, 1.0);
    let y = point.y.clamp(0.0, 1.0);
    let z = point.z.clamp(0.0, 1.0);
    let dx = ((values[1] - values[0]) * (1.0 - y) * (1.0 - z)
        + (values[3] - values[2]) * y * (1.0 - z)
        + (values[5] - values[4]) * (1.0 - y) * z
        + (values[7] - values[6]) * y * z)
        / voxel_size;
    let dy = ((values[2] - values[0]) * (1.0 - x) * (1.0 - z)
        + (values[3] - values[1]) * x * (1.0 - z)
        + (values[6] - values[4]) * (1.0 - x) * z
        + (values[7] - values[5]) * x * z)
        / voxel_size;
    let dz = ((values[4] - values[0]) * (1.0 - x) * (1.0 - y)
        + (values[5] - values[1]) * x * (1.0 - y)
        + (values[6] - values[2]) * (1.0 - x) * y
        + (values[7] - values[3]) * x * y)
        / voxel_size;
    DVec3::new(dx, dy, dz)
}

fn signed_distance(point: DVec3, triangles: &[[DVec3; 3]]) -> f64 {
    let mut closest_squared = f64::INFINITY;
    let mut intersections = 0_usize;
    let ray = DVec3::new(1.0, 0.371_390_676_354_103_7, 0.529_173_248_214_697_2);
    for triangle in triangles {
        closest_squared = closest_squared.min(point_triangle_distance_squared(point, *triangle));
        if ray_intersects_triangle(point, ray, *triangle) {
            intersections += 1;
        }
    }
    let distance = closest_squared.sqrt();
    if intersections % 2 == 1 {
        -distance
    } else {
        distance
    }
}

#[expect(
    clippy::many_single_char_names,
    reason = "point-triangle distance uses standard barycentric variables"
)]
fn point_triangle_distance_squared(point: DVec3, [a, b, c]: [DVec3; 3]) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length_squared();
    }
    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length_squared();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (point - (a + v * ab)).length_squared();
    }
    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length_squared();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (point - (a + w * ac)).length_squared();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (point - (b + w * (c - b))).length_squared();
    }
    let denominator = 1.0 / (va + vb + vc);
    let v = vb * denominator;
    let w = vc * denominator;
    (point - (a + ab * v + ac * w)).length_squared()
}

#[expect(
    clippy::many_single_char_names,
    reason = "ray-triangle intersection uses standard vector notation"
)]
fn ray_intersects_triangle(origin: DVec3, direction: DVec3, [a, b, c]: [DVec3; 3]) -> bool {
    let edge1 = b - a;
    let edge2 = c - a;
    let p = direction.cross(edge2);
    let determinant = edge1.dot(p);
    if determinant.abs() <= f64::MIN_POSITIVE {
        return false;
    }
    let inverse = 1.0 / determinant;
    let offset = origin - a;
    let u = offset.dot(p) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = offset.cross(edge1);
    let v = direction.dot(q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    edge2.dot(q) * inverse > 0.0
}

fn add_oriented_triangle(
    mesh: &mut Mesh,
    edge_keys: &mut HashSet<(u32, u32)>,
    first: (u32, DVec3),
    second: (u32, DVec3),
    third: (u32, DVec3),
    outward: DVec3,
) -> Result<()> {
    if first.0 == second.0 || second.0 == third.0 || third.0 == first.0 {
        return Ok(());
    }
    let normal = (second.1 - first.1).cross(third.1 - first.1);
    if normal.length_squared() <= f64::MIN_POSITIVE {
        return Ok(());
    }
    if mesh.faces.len() >= MAX_OUTPUT_FACES {
        return Err(resource_error());
    }
    let vertices = if normal.dot(outward) < 0.0 {
        [first.0, third.0, second.0]
    } else {
        [first.0, second.0, third.0]
    };
    let boundary = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[0]],
    ];
    let mut missing_edges = [[0_u32; 2]; 3];
    let mut missing_count = 0;
    for edge in boundary {
        if !edge_keys.contains(&edge_key(edge[0], edge[1])) {
            missing_edges[missing_count] = edge;
            missing_count += 1;
        }
    }
    let edge_count = u32::try_from(missing_count).map_err(|_| resource_error())?;
    let next_edge = mesh
        .next_id
        .edge
        .checked_add(edge_count)
        .ok_or_else(|| mesh_output_error(&MeshError::IdExhausted))?;
    let face_id = mesh.next_id.face;
    let next_face = face_id
        .checked_add(1)
        .ok_or_else(|| mesh_output_error(&MeshError::IdExhausted))?;

    mesh.edges.reserve(missing_count);
    mesh.faces.reserve(1);
    edge_keys.reserve(missing_count);
    for (edge_id, edge) in (mesh.next_id.edge..next_edge).zip(&missing_edges[..missing_count]) {
        mesh.edges.push(Edge {
            id: edge_id,
            vertices: *edge,
        });
        edge_keys.insert(edge_key(edge[0], edge[1]));
    }
    mesh.next_id.edge = next_edge;
    mesh.faces.push(Face {
        id: face_id,
        vertices: vertices.to_vec(),
        material_index: 0,
    });
    mesh.next_id.face = next_face;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]
    use std::collections::HashMap;

    use glam::DVec3;
    use proptest::prelude::*;

    use crate::geom::{BoxParams, Mesh, PlaneParams};

    use super::remesh;

    fn assert_watertight(mesh: &Mesh) {
        let mut incidence = HashMap::new();
        for face in &mesh.faces {
            assert_eq!(face.vertices.len(), 3);
            for index in 0..face.vertices.len() {
                let a = face.vertices[index];
                let b = face.vertices[(index + 1) % face.vertices.len()];
                *incidence
                    .entry(if a < b { (a, b) } else { (b, a) })
                    .or_insert(0) += 1;
            }
        }
        assert!(!incidence.is_empty());
        assert!(incidence.values().all(|&count| count == 2));
        assert_eq!(mesh.edges.len(), incidence.len());
    }

    fn assert_bounds_close(source: &Mesh, result: &Mesh, voxel_size: f64) {
        let source_bounds = source.bounds().unwrap();
        let result_bounds = result.bounds().unwrap();
        for axis in 0..3 {
            assert!((source_bounds.min[axis] - result_bounds.min[axis]).abs() <= voxel_size);
            assert!((source_bounds.max[axis] - result_bounds.max[axis]).abs() <= voxel_size);
        }
    }

    #[test]
    fn box_remesh_is_watertight_and_preserves_bounds_within_voxel() {
        let source = Mesh::box_mesh(BoxParams {
            size: DVec3::new(1.7, 2.1, 1.3),
        })
        .unwrap();
        let result = remesh(&source, 0.3).unwrap();
        assert_watertight(&result);
        assert_bounds_close(&source, &result, 0.3);
    }

    proptest! {
        #[test]
        fn closed_box_remesh_keeps_watertight_bounds(
            x in 0.7_f64..2.4,
            y in 0.7_f64..2.4,
            z in 0.7_f64..2.4,
        ) {
            let source = Mesh::box_mesh(BoxParams { size: DVec3::new(x, y, z) }).unwrap();
            let voxel_size = 0.35;
            let result = remesh(&source, voxel_size).unwrap();
            assert_watertight(&result);
            assert_bounds_close(&source, &result, voxel_size);
        }
    }

    #[test]
    fn voxel_size_must_be_finite_and_positive() {
        let source = Mesh::box_mesh(BoxParams::default()).unwrap();
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(remesh(&source, invalid).is_err());
        }
    }
    #[test]
    fn rejects_open_meshes_and_excessive_grid_resolution() {
        let plane = Mesh::plane(PlaneParams::default()).unwrap();
        assert!(remesh(&plane, 0.2).is_err());

        let closed = Mesh::box_mesh(BoxParams::default()).unwrap();
        assert!(remesh(&closed, 1.0e-6).is_err());
    }
}
