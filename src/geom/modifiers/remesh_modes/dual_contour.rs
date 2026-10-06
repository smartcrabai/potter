// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   intern/dualcon/intern/octree.cpp, SPDX-FileCopyrightText: 2011-2023 Blender Authors, GPL-2.0-or-later.
//   intern/dualcon/intern/Projections.cpp, SPDX-FileCopyrightText: 2002-2022 Blender Authors, GPL-2.0-or-later.

//! Uniform-depth dual-contouring implementation used by the octree Remesh modes.

use glam::{DVec3, Vec3};
use serde_json::json;
use std::collections::HashMap;

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
};

const GRID_DIMENSION: i64 = 1 << 20;
const CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
];
const EDGES: [[usize; 2]; 12] = [
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
];

#[derive(Clone, Copy)]
struct GridIntersection {
    offset: f32,
    normal: [f32; 3],
}

struct ProjectionTriangle {
    axes: [[i64; 3]; 13],
    projections: [[i64; 2]; 13],
    normal: [f32; 3],
    minimum: [i64; 3],
    maximum: [i64; 3],
}

impl ProjectionTriangle {
    fn new(points: [[i64; 3]; 3]) -> Option<Self> {
        let mut axes = [[0_i64; 3]; 13];
        axes[0] = [1, 0, 0];
        axes[1] = [0, 1, 0];
        axes[2] = [0, 0, 1];
        let edges = [
            subtract(points[1], points[0]),
            subtract(points[2], points[1]),
            subtract(points[0], points[2]),
        ];
        axes[3] = cross(edges[0], edges[1]);
        let normal_length = axes[3]
            .iter()
            .map(|component| (*component as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        if normal_length <= 0.0 {
            return None;
        }
        let normal = axes[3].map(|component| (component as f64 / normal_length) as f32);
        for (edge_index, edge) in edges.iter().enumerate() {
            for axis_index in 0..3 {
                axes[4 + edge_index * 3 + axis_index] = cross(axes[axis_index], *edge);
            }
        }

        let mut projections = [[0_i64; 2]; 13];
        for axis_index in 0..13 {
            let first = dot(axes[axis_index], points[0]);
            let second = dot(axes[axis_index], points[1]);
            let third = dot(axes[axis_index], points[2]);
            projections[axis_index] = [first.min(second).min(third), first.max(second).max(third)];
        }
        let mut minimum = points[0];
        let mut maximum = points[0];
        for point in &points[1..] {
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(point[axis]);
                maximum[axis] = maximum[axis].max(point[axis]);
            }
        }
        Some(Self {
            axes,
            projections,
            normal,
            minimum,
            maximum,
        })
    }

    fn overlaps_octree_path(&self, cell: [usize; 3], depth: u32) -> bool {
        let mut origin = [0_i64; 3];
        let mut span = GRID_DIMENSION;
        for bit in (0..depth).rev() {
            let half_span = span / 2;
            for axis in 0..3 {
                let midpoint = origin[axis] + half_span;
                if (cell[axis] >> bit) & 1 == 0 {
                    if midpoint < self.minimum[axis] {
                        return false;
                    }
                } else {
                    if midpoint >= self.maximum[axis] {
                        return false;
                    }
                    origin[axis] += half_span;
                }
            }
            span = half_span;
        }
        true
    }
    fn intersects_cube(&self, origin: [i64; 3], length: i64) -> bool {
        for axis in 0..13 {
            let projection_origin = dot(self.axes[axis], origin);
            let mut cube_min = projection_origin;
            let mut cube_max = projection_origin;
            for coordinate in 0..3 {
                let edge_projection = self.axes[axis][coordinate] * length;
                cube_min += edge_projection.min(0);
                cube_max += edge_projection.max(0);
            }
            let [triangle_min, triangle_max] = self.projections[axis];
            if cube_min > triangle_max || cube_max < triangle_min {
                return false;
            }
        }
        true
    }

    fn intersects_primary(&self, start: [i64; 3], direction: usize, length: i64) -> Option<f32> {
        for axis in 0..13 {
            let projection_start = dot(self.axes[axis], start);
            let projection_end = projection_start + self.axes[axis][direction] * length;
            let [triangle_min, triangle_max] = self.projections[axis];
            if projection_start < projection_end {
                if projection_start > triangle_max || projection_end < triangle_min {
                    return None;
                }
            } else if projection_end > triangle_max || projection_start < triangle_min {
                return None;
            }
        }

        let normal_axis = self.axes[3];
        let projection_start = dot(normal_axis, start);
        let projection_end = projection_start + normal_axis[direction] * length;
        let denominator = projection_end - projection_start;
        let offset = if denominator == 0 {
            0.5
        } else {
            let value = (self.projections[3][1] - projection_start) as f64 / denominator as f64;
            if (0.0..=1.0).contains(&value) {
                value as f32
            } else {
                0.5
            }
        };
        Some(offset)
    }
}
pub(super) fn is_closed_two_manifold(mesh: &Mesh) -> bool {
    if mesh.faces.is_empty() {
        return false;
    }
    let mut edge_uses = HashMap::<(u32, u32), u8>::new();
    for face in &mesh.faces {
        if face.vertices.len() < 3 {
            return false;
        }
        for index in 0..face.vertices.len() {
            let first = face.vertices[index];
            let second = face.vertices[(index + 1) % face.vertices.len()];
            let count = edge_uses
                .entry((first.min(second), first.max(second)))
                .or_default();
            *count = count.saturating_add(1);
        }
    }
    !edge_uses.is_empty() && edge_uses.values().all(|uses| *uses == 2)
}
#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
pub(super) fn remesh(
    mesh: &Mesh,
    depth: u32,
    scale: f64,
    mode: &str,
    sharpness: f64,
    modifier_id: &crate::model::Id,
) -> Result<Mesh> {
    if !matches!(mode, "BLOCKS" | "SMOOTH" | "SHARP") {
        return Err(PotError::invalid_operation("unsupported dual contour mode"));
    }

    let triangles = blender_corner_triangles(mesh)?;
    if triangles.is_empty() {
        return Err(PotError::invalid_operation(
            "dual contour remesh requires target polygons",
        ));
    }
    let bounds = mesh.bounds().ok_or_else(|| {
        PotError::invalid_operation("dual contour remesh requires a non-empty mesh")
    })?;
    let source_min = bounds.min.to_array().map(|value| value as f32);
    let source_max = bounds.max.to_array().map(|value| value as f32);
    let mut range = 0.0_f32;
    for axis in 0..3 {
        range = range.max(source_max[axis] - source_min[axis]);
    }
    if !range.is_finite() || range <= 0.0 {
        return Err(PotError::invalid_operation(
            "dual contour remesh requires non-zero bounds",
        ));
    }

    let mut origin = source_min;
    for axis in 0..3 {
        origin[axis] = (source_max[axis] + source_min[axis]) / 2.0 - range / 2.0;
    }
    let scale = scale as f32;
    let inverse_scale = 1.0_f32 / scale;
    for coordinate in &mut origin {
        *coordinate -= range * (inverse_scale - 1.0) / 2.0;
    }
    range *= inverse_scale;

    let cells = 1_usize.checked_shl(depth).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "dual contour depth exceeds grid limits",
        )
    })?;
    let side = cells + 1;
    let node_count = side
        .checked_mul(side)
        .and_then(|value| value.checked_mul(side))
        .ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "dual contour grid exceeds limits")
        })?;
    let cell_count = cells
        .checked_mul(cells)
        .and_then(|value| value.checked_mul(cells))
        .ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "dual contour cells exceed limits")
        })?;
    if node_count > 300_000 || cell_count > 270_000 {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            format!("dual contour grid requires {node_count} nodes and {cell_count} cells"),
        ));
    }

    let cell_span = GRID_DIMENSION / cells as i64;
    let projection_triangles = triangles
        .iter()
        .filter_map(|triangle| {
            let points = triangle.map(|index| {
                let coordinate = mesh.vertices[index].co.to_array();
                std::array::from_fn(|axis| {
                    let projected =
                        GRID_DIMENSION as f32 * (coordinate[axis] as f32 - origin[axis]) / range;
                    projected as i64
                })
            });
            ProjectionTriangle::new(points)
        })
        .collect::<Vec<_>>();
    if projection_triangles.is_empty() {
        return Err(PotError::invalid_operation(
            "dual contour remesh requires non-degenerate target triangles",
        ));
    }

    let (intersections, active_cells) =
        scan_convert_edges(&projection_triangles, cells, side, cell_span, depth);
    let mut cell_signs = vec![0_u8; cell_count];
    let sign_table = dualcon_sign_table();
    build_sign_masks(
        &sign_table,
        &intersections,
        &mut cell_signs,
        [0, 0, 0],
        cells,
        depth,
        1,
        cells,
        side,
    );
    let paths = find_paths(&intersections, &active_cells, cells, side);
    let (path_parents, path_degrees) = combine_paths(&paths, cell_count);
    let ring_cells = connect_paths_to_ring(&path_parents, &path_degrees);
    let signs = scanline_node_signs(&intersections, cells, side);
    let closed_surface = is_closed_two_manifold(mesh);
    if !closed_surface && has_unpatched_sign_rings(&cell_signs, &signs, &ring_cells, cells, side) {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Blender dual-contouring ring patch could not close the input surface",
            json!({
                "feature_id":"modifier.remesh.open_nonmanifold",
                "modifier_id":modifier_id,
                "mode":mode
            }),
        ));
    }
    patch_incomplete_sign_rings(
        &mut cell_signs,
        &signs,
        &ring_cells,
        closed_surface,
        cells,
        side,
    );

    let mut cell_vertices = vec![u32::MAX; cell_count];
    let mut output_vertices = Vec::<DVec3>::new();
    for morton in 0..cell_count {
        let [x, y, z] = morton_cell(morton, depth);
        let coordinates = CORNERS.map(|corner| [x + corner[0], y + corner[1], z + corner[2]]);
        let cell = cell_index(x, y, z, cells);
        let sign_mask = cell_signs[cell];
        if sign_mask == 0 || sign_mask == u8::MAX {
            continue;
        }

        let mut hermite_points = [DVec3::ZERO; 12];
        let mut hermite_normals = [DVec3::ZERO; 12];
        let mut intersection_count = 0;
        for edge in EDGES {
            let start = coordinates[edge[0]];
            let direction = (0..3)
                .find(|axis| start[*axis] != coordinates[edge[1]][*axis])
                .ok_or_else(|| PotError::invalid_operation("dual contour edge has no direction"))?;
            let cache_index = edge_index_for(direction, start[0], start[1], start[2], cells, side);
            let Some(intersection) = intersections[direction][cache_index] else {
                continue;
            };
            let mut point = start.map(|coordinate| coordinate as f32 * cell_span as f32);
            point[direction] += intersection.offset * cell_span as f32;
            hermite_points[intersection_count] = DVec3::from_array(point.map(f64::from));
            hermite_normals[intersection_count] =
                DVec3::from_array(intersection.normal.map(f64::from));
            intersection_count += 1;
        }

        if intersection_count == 0 {
            continue;
        }
        let mass_point = hermite_points[..intersection_count]
            .iter()
            .copied()
            .sum::<DVec3>()
            / intersection_count as f64;
        let cell_minimum = DVec3::new(
            (x as i64 * cell_span) as f64,
            (y as i64 * cell_span) as f64,
            (z as i64 * cell_span) as f64,
        );
        let cell_maximum = cell_minimum + DVec3::splat(cell_span as f64);
        let grid_position = match mode {
            "BLOCKS" => cell_minimum + DVec3::splat((cell_span / 2) as f64),
            "SMOOTH" => mass_point,
            "SHARP" => {
                let (minimizer, sharp_mass_point) = qef_vertex(
                    &hermite_points[..intersection_count],
                    &hermite_normals[..intersection_count],
                );
                let margin = DVec3::splat(sharpness * cell_span as f64);
                let lower = cell_minimum - margin;
                let upper = cell_maximum + margin;
                if (minimizer - lower).min_element() < 0.0
                    || (minimizer - upper).max_element() > 0.0
                {
                    sharp_mass_point
                } else {
                    minimizer
                }
            }
            _ => unreachable!(),
        };
        let position = DVec3::from_array(std::array::from_fn(|axis| {
            (grid_position[axis] as f32 * range / GRID_DIMENSION as f32) as f64
                + origin[axis] as f64
        }));
        let vertex_index = u32::try_from(output_vertices.len()).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "dual contour output exceeds vertex limits",
            )
        })?;
        cell_vertices[cell_index(x, y, z, cells)] = vertex_index;
        output_vertices.push(position);
    }

    let mut polygons = Vec::new();
    for z in 1..cells {
        for y in 1..cells {
            for x in 0..cells {
                let first = node_index(x, y, z, side);
                let second = node_index(x + 1, y, z, side);
                if signs[first] != signs[second] {
                    add_quad(
                        &mut polygons,
                        &cell_vertices,
                        [
                            cell_index(x, y - 1, z - 1, cells),
                            cell_index(x, y, z - 1, cells),
                            cell_index(x, y, z, cells),
                            cell_index(x, y - 1, z, cells),
                        ],
                        signs[first] == 1,
                    );
                }
            }
        }
    }
    for z in 1..cells {
        for y in 0..cells {
            for x in 1..cells {
                let first = node_index(x, y, z, side);
                let second = node_index(x, y + 1, z, side);
                if signs[first] != signs[second] {
                    add_quad(
                        &mut polygons,
                        &cell_vertices,
                        [
                            cell_index(x - 1, y, z - 1, cells),
                            cell_index(x - 1, y, z, cells),
                            cell_index(x, y, z, cells),
                            cell_index(x, y, z - 1, cells),
                        ],
                        signs[first] == 1,
                    );
                }
            }
        }
    }
    for z in 0..cells {
        for y in 1..cells {
            for x in 1..cells {
                let first = node_index(x, y, z, side);
                let second = node_index(x, y, z + 1, side);
                if signs[first] != signs[second] {
                    add_quad(
                        &mut polygons,
                        &cell_vertices,
                        [
                            cell_index(x - 1, y - 1, z, cells),
                            cell_index(x, y - 1, z, cells),
                            cell_index(x, y, z, cells),
                            cell_index(x - 1, y, z, cells),
                        ],
                        signs[first] == 1,
                    );
                }
            }
        }
    }
    if polygons.is_empty() {
        return Err(PotError::invalid_operation(
            "dual contour remesh produced no surface",
        ));
    }
    Mesh::from_positions_and_faces(output_vertices, polygons).map_err(|error| {
        PotError::invalid_argument(format!("dual contour mesh construction failed: {error}"))
    })
}

fn blender_corner_triangles(mesh: &Mesh) -> Result<Vec<[usize; 3]>> {
    let mut vertex_indices = HashMap::with_capacity(mesh.vertices.len());
    for (index, vertex) in mesh.vertices.iter().enumerate() {
        vertex_indices.insert(vertex.id, index);
    }

    let mut triangles = Vec::new();
    for face in &mesh.faces {
        let vertices = face
            .vertices
            .iter()
            .map(|vertex| vertex_indices.get(vertex).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                PotError::invalid_operation("dual contour face references an unknown vertex")
            })?;
        if vertices.len() == 4 && blender_quad_uses_alternate_diagonal(mesh, &vertices) {
            triangles.push([vertices[0], vertices[1], vertices[3]]);
            triangles.push([vertices[1], vertices[2], vertices[3]]);
        } else {
            for corner in 1..vertices.len().saturating_sub(1) {
                triangles.push([vertices[0], vertices[corner], vertices[corner + 1]]);
            }
        }
    }
    Ok(triangles)
}

fn blender_quad_uses_alternate_diagonal(mesh: &Mesh, vertices: &[usize]) -> bool {
    let points: [Vec3; 4] = std::array::from_fn(|corner| {
        Vec3::from_array(
            mesh.vertices[vertices[corner]]
                .co
                .to_array()
                .map(|coordinate| coordinate as f32),
        )
    });
    let first_normal = (points[1] - points[0]).cross(points[2] - points[0]);
    let second_normal = (points[3] - points[0]).cross(points[2] - points[0]);
    first_normal.dot(second_normal) > 0.0
}
fn scan_convert_edges(
    triangles: &[ProjectionTriangle],
    cells: usize,
    side: usize,
    cell_span: i64,
    depth: u32,
) -> ([Vec<Option<GridIntersection>>; 3], Vec<bool>) {
    let mut intersections = [
        vec![None; cells * side * side],
        vec![None; cells * side * side],
        vec![None; cells * side * side],
    ];
    let mut active_cells = vec![false; cells * cells * cells];
    for triangle in triangles {
        let cell_ranges: [(usize, usize); 3] = std::array::from_fn(|axis| {
            grid_range(
                triangle.minimum[axis],
                triangle.maximum[axis],
                cell_span,
                cells - 1,
                true,
            )
        });
        for z in cell_ranges[2].0..=cell_ranges[2].1 {
            for y in cell_ranges[1].0..=cell_ranges[1].1 {
                for x in cell_ranges[0].0..=cell_ranges[0].1 {
                    let cell = [x, y, z];
                    let origin = [
                        x as i64 * cell_span,
                        y as i64 * cell_span,
                        z as i64 * cell_span,
                    ];
                    if triangle.overlaps_octree_path(cell, depth)
                        && triangle.intersects_cube(origin, cell_span)
                    {
                        active_cells[cell_index(x, y, z, cells)] = true;
                    }
                }
            }
        }
    }
    for triangle in triangles {
        for direction in 0..3 {
            let first_perpendicular = (direction + 1) % 3;
            let second_perpendicular = (direction + 2) % 3;
            let along = grid_range(
                triangle.minimum[direction],
                triangle.maximum[direction],
                cell_span,
                cells - 1,
                true,
            );
            let first = grid_range(
                triangle.minimum[first_perpendicular],
                triangle.maximum[first_perpendicular],
                cell_span,
                cells,
                false,
            );
            let second = grid_range(
                triangle.minimum[second_perpendicular],
                triangle.maximum[second_perpendicular],
                cell_span,
                cells,
                false,
            );
            for index in along.0..=along.1 {
                for first_index in first.0..=first.1 {
                    for second_index in second.0..=second.1 {
                        let mut start = [0_i64; 3];
                        start[direction] = index as i64 * cell_span;
                        start[first_perpendicular] = first_index as i64 * cell_span;
                        start[second_perpendicular] = second_index as i64 * cell_span;
                        let cache_index = edge_index_for(
                            direction,
                            start[0] as usize / cell_span as usize,
                            start[1] as usize / cell_span as usize,
                            start[2] as usize / cell_span as usize,
                            cells,
                            side,
                        );
                        if intersections[direction][cache_index].is_some() {
                            continue;
                        }
                        let cell = [
                            start[0] as usize / cell_span as usize,
                            start[1] as usize / cell_span as usize,
                            start[2] as usize / cell_span as usize,
                        ];
                        if !triangle.overlaps_octree_path(cell, depth) {
                            continue;
                        }
                        if triangle.intersects_cube(start, cell_span)
                            && let Some(offset) =
                                triangle.intersects_primary(start, direction, cell_span)
                        {
                            intersections[direction][cache_index] = Some(GridIntersection {
                                offset,
                                normal: triangle.normal,
                            });
                        }
                    }
                }
            }
        }
    }
    (intersections, active_cells)
}

fn grid_range(
    minimum: i64,
    maximum: i64,
    cell_span: i64,
    last_index: usize,
    segment: bool,
) -> (usize, usize) {
    let first = minimum.div_euclid(cell_span).saturating_sub(1).max(0) as usize;
    let extra = i64::from(!segment);
    let last = maximum
        .div_euclid(cell_span)
        .saturating_add(extra)
        .min(last_index as i64) as usize;
    (first.min(last), first.max(last))
}

fn dualcon_sign_table() -> [u8; 1 << 12] {
    let mut table = [0_u8; 1 << 12];
    for sign_mask in 0..=u8::MAX {
        let mut edge_mask = 0_usize;
        for edge in (0..EDGES.len()).rev() {
            edge_mask <<= 1;
            let [first, second] = EDGES[edge];
            if ((sign_mask >> first) ^ (sign_mask >> second)) & 1 != 0 {
                edge_mask |= 1;
            }
        }
        table[edge_mask] = sign_mask;
    }
    table
}

#[expect(
    clippy::manual_memcpy,
    reason = "recursive octree sign propagation selects one corner from each child result"
)]
fn build_sign_masks(
    table: &[u8; 1 << 12],
    intersections: &[Vec<Option<GridIntersection>>; 3],
    cell_signs: &mut [u8],
    origin: [usize; 3],
    span: usize,
    depth: u32,
    start: u8,
    cells: usize,
    side: usize,
) -> [u8; 8] {
    if depth == 0 {
        let mut edge_mask = 0_usize;
        for edge_index in 0..EDGES.len() {
            let corner = CORNERS[EDGES[edge_index][0]];
            let point: [usize; 3] = std::array::from_fn(|axis| origin[axis] + corner[axis]);
            let direction = edge_index / 4;
            let cache_index = edge_index_for(direction, point[0], point[1], point[2], cells, side);
            if intersections[direction][cache_index].is_some() {
                edge_mask |= 1 << edge_index;
            }
        }
        let mut sign_mask = table[edge_mask];
        if ((start ^ sign_mask) & 1) != 0 {
            sign_mask = !sign_mask;
        }
        cell_signs[cell_index(origin[0], origin[1], origin[2], cells)] = sign_mask;
        return std::array::from_fn(|corner| (sign_mask >> corner) & 1);
    }

    let child_span = span / 2;
    let mut signs = [0_u8; 8];
    signs[0] = start;
    let first_child = build_sign_masks(
        table,
        intersections,
        cell_signs,
        origin,
        child_span,
        depth - 1,
        start,
        cells,
        side,
    );
    for child in 1..8 {
        let child_origin =
            std::array::from_fn(|axis| origin[axis] + CORNERS[child][axis] * child_span);
        let child_signs = build_sign_masks(
            table,
            intersections,
            cell_signs,
            child_origin,
            child_span,
            depth - 1,
            first_child[child],
            cells,
            side,
        );
        signs[child] = child_signs[child];
    }
    signs
}

fn cell_edge_mask(
    intersections: &[Vec<Option<GridIntersection>>; 3],
    x: usize,
    y: usize,
    z: usize,
    cells: usize,
    side: usize,
) -> u16 {
    let mut mask = 0_u16;
    for edge in 0..EDGES.len() {
        let corner = CORNERS[EDGES[edge][0]];
        let start = [x + corner[0], y + corner[1], z + corner[2]];
        let direction = edge / 4;
        let index = edge_index_for(direction, start[0], start[1], start[2], cells, side);
        if intersections[direction][index].is_some() {
            mask |= 1 << edge;
        }
    }
    mask
}

fn find_paths(
    intersections: &[Vec<Option<GridIntersection>>; 3],
    active_cells: &[bool],
    cells: usize,
    side: usize,
) -> Vec<(usize, usize)> {
    const HIGH_FACE_EDGES: [[usize; 4]; 3] = [[6, 10, 7, 11], [2, 9, 3, 11], [1, 5, 3, 7]];
    let mut paths = Vec::new();
    for z in 0..cells {
        for y in 0..cells {
            for x in 0..cells {
                let cell = cell_index(x, y, z, cells);
                let edge_mask = cell_edge_mask(intersections, x, y, z, cells, side);
                if !active_cells[cell] {
                    continue;
                }
                for direction in 0..3 {
                    let coordinate = [x, y, z];
                    if coordinate[direction] + 1 >= cells
                        || (0..3).any(|axis| axis != direction && coordinate[axis] == cells)
                    {
                        continue;
                    }
                    let face_parity = HIGH_FACE_EDGES[direction]
                        .iter()
                        .filter(|edge| edge_mask & (1 << **edge) != 0)
                        .count()
                        & 1;
                    if face_parity == 0 {
                        continue;
                    }
                    let mut neighbor = coordinate;
                    neighbor[direction] += 1;
                    let neighbor_index = cell_index(neighbor[0], neighbor[1], neighbor[2], cells);
                    if active_cells[neighbor_index] {
                        paths.push((cell, neighbor_index));
                    }
                }
            }
        }
    }
    paths
}

fn combine_paths(paths: &[(usize, usize)], cell_count: usize) -> (Vec<usize>, Vec<u8>) {
    let mut parents = (0..cell_count).collect::<Vec<_>>();
    let mut degrees = vec![0_u8; cell_count];
    for &(first, second) in paths {
        degrees[first] = degrees[first].saturating_add(1);
        degrees[second] = degrees[second].saturating_add(1);
        let first_root = find_path_root(&mut parents, first);
        let second_root = find_path_root(&mut parents, second);
        if first_root != second_root {
            parents[second_root] = first_root;
        }
    }
    (parents, degrees)
}

fn connect_paths_to_ring(parents: &[usize], degrees: &[u8]) -> Vec<bool> {
    let mut roots = (0..parents.len()).collect::<Vec<_>>();
    for index in 0..roots.len() {
        roots[index] = find_path_root(&mut roots, index);
    }
    let mut component_sizes = vec![0_usize; parents.len()];
    let mut component_degree_sums = vec![0_usize; parents.len()];
    for index in 0..degrees.len() {
        if degrees[index] == 0 {
            continue;
        }
        let root = roots[index];
        component_sizes[root] += 1;
        component_degree_sums[root] += usize::from(degrees[index]);
    }
    let mut ring_cells = vec![false; degrees.len()];
    for index in 0..degrees.len() {
        if degrees[index] == 0 {
            continue;
        }
        let root = roots[index];
        if component_sizes[root] >= 3 && component_degree_sums[root] >= component_sizes[root] * 2 {
            ring_cells[index] = true;
        }
    }
    ring_cells
}

fn find_path_root(parents: &mut [usize], mut index: usize) -> usize {
    while parents[index] != index {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    index
}
fn has_unpatched_sign_rings(
    cell_signs: &[u8],
    node_signs: &[u8],
    ring_cells: &[bool],
    cells: usize,
    side: usize,
) -> bool {
    for z in 0..cells {
        for y in 0..cells {
            for x in 0..cells {
                let propagated = CORNERS
                    .iter()
                    .enumerate()
                    .fold(0_u8, |mask, (corner, offset)| {
                        let node = node_index(x + offset[0], y + offset[1], z + offset[2], side);
                        mask | ((node_signs[node] & 1) << corner)
                    });
                let cell = cell_index(x, y, z, cells);
                if (cell_signs[cell] == 0 || cell_signs[cell] == u8::MAX)
                    && propagated != 0
                    && propagated != u8::MAX
                    && !ring_cells[cell]
                {
                    return true;
                }
            }
        }
    }
    false
}
fn patch_incomplete_sign_rings(
    cell_signs: &mut [u8],
    node_signs: &[u8],
    ring_cells: &[bool],
    closed_surface: bool,
    cells: usize,
    side: usize,
) {
    for z in 0..cells {
        for y in 0..cells {
            for x in 0..cells {
                let propagated = CORNERS
                    .iter()
                    .enumerate()
                    .fold(0_u8, |mask, (corner, offset)| {
                        let node = node_index(x + offset[0], y + offset[1], z + offset[2], side);
                        mask | ((node_signs[node] & 1) << corner)
                    });
                let cell = cell_index(x, y, z, cells);
                if (cell_signs[cell] == 0 || cell_signs[cell] == u8::MAX)
                    && propagated != 0
                    && propagated != u8::MAX
                    && (ring_cells[cell] || closed_surface)
                {
                    cell_signs[cell] = propagated;
                }
            }
        }
    }
}

fn scanline_node_signs(
    intersections: &[Vec<Option<GridIntersection>>; 3],
    cells: usize,
    side: usize,
) -> Vec<u8> {
    let mut signs = vec![1_u8; side * side * side];
    for z in 0..side {
        for y in 0..side {
            let mut sign = 1_u8;
            for x in 0..cells {
                signs[node_index(x, y, z, side)] = sign;
                let edge = edge_index_for(0, x, y, z, cells, side);
                sign ^= u8::from(intersections[0][edge].is_some());
            }
            signs[node_index(cells, y, z, side)] = sign;
        }
    }
    signs
}
fn edge_index_for(
    direction: usize,
    x: usize,
    y: usize,
    z: usize,
    cells: usize,
    side: usize,
) -> usize {
    match direction {
        0 => (z * side + y) * cells + x,
        1 => (z * cells + y) * side + x,
        _ => (z * side + y) * side + x,
    }
}

fn node_index(x: usize, y: usize, z: usize, side: usize) -> usize {
    z * side * side + y * side + x
}

fn cell_index(x: usize, y: usize, z: usize, cells: usize) -> usize {
    z * cells * cells + y * cells + x
}

fn subtract(left: [i64; 3], right: [i64; 3]) -> [i64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn cross(left: [i64; 3], right: [i64; 3]) -> [i64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [i64; 3], right: [i64; 3]) -> i64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn add_quad(
    polygons: &mut Vec<Vec<usize>>,
    cell_vertices: &[u32],
    cell_indices: [usize; 4],
    reverse: bool,
) {
    let mut vertices = [0_usize; 4];
    for (index, cell) in cell_indices.into_iter().enumerate() {
        let vertex = cell_vertices[cell];
        if vertex == u32::MAX {
            return;
        }
        vertices[index] = vertex as usize;
    }
    if vertices[0] == vertices[1]
        || vertices[1] == vertices[2]
        || vertices[2] == vertices[3]
        || vertices[3] == vertices[0]
        || vertices[0] == vertices[2]
        || vertices[1] == vertices[3]
    {
        return;
    }
    if reverse {
        vertices.reverse();
    }
    polygons.push(vertices.to_vec());
}

fn qef_vertex(points: &[DVec3], normals: &[DVec3]) -> (DVec3, DVec3) {
    let mut matrix = [[0.0_f32; 3]; 3];
    let mut rhs = [0.0_f32; 3];
    let mut midpoint = [0.0_f32; 3];
    for (point, normal) in points.iter().zip(normals) {
        let point = point.to_array().map(|coordinate| coordinate as f32);
        let normal = normal.to_array().map(|coordinate| coordinate as f32);
        let projected = point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2];
        for row in 0..3 {
            rhs[row] += normal[row] * projected;
            midpoint[row] += point[row];
            for column in 0..3 {
                matrix[row][column] += normal[row] * normal[column];
            }
        }
    }
    for coordinate in &mut midpoint {
        *coordinate /= points.len() as f32;
    }

    let mut residual = [0.0_f32; 3];
    for row in 0..3 {
        residual[row] = rhs[row]
            - (matrix[row][0] * midpoint[0]
                + matrix[row][1] * midpoint[1]
                + matrix[row][2] * midpoint[2]);
    }

    let mut eigenvectors = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..24 {
        let mut pivot = (0, 1);
        for (left, right) in [(0, 2), (1, 2)] {
            if matrix[left][right].abs() > matrix[pivot.0][pivot.1].abs() {
                pivot = (left, right);
            }
        }
        let (left, right) = pivot;
        let off_diagonal = matrix[left][right];
        if off_diagonal.abs() <= 1.0e-7 {
            break;
        }
        let tau = (matrix[right][right] - matrix[left][left]) / (2.0 * off_diagonal);
        let tangent = if tau >= 0.0 {
            1.0 / (tau + tau.hypot(1.0))
        } else {
            -1.0 / (-tau + tau.hypot(1.0))
        };
        let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
        let sine = tangent * cosine;
        let left_diagonal = matrix[left][left];
        let right_diagonal = matrix[right][right];
        matrix[left][left] = left_diagonal - tangent * off_diagonal;
        matrix[right][right] = right_diagonal + tangent * off_diagonal;
        matrix[left][right] = 0.0;
        matrix[right][left] = 0.0;
        let mut left_values = [0.0_f32; 3];
        let mut right_values = [0.0_f32; 3];
        for (index, row) in matrix.iter_mut().enumerate() {
            if index == left || index == right {
                continue;
            }
            let left_value = row[left];
            let right_value = row[right];
            let updated_left = cosine * left_value - sine * right_value;
            let updated_right = sine * left_value + cosine * right_value;
            row[left] = updated_left;
            row[right] = updated_right;
            left_values[index] = updated_left;
            right_values[index] = updated_right;
        }
        for (index, (value, updated)) in matrix[left].iter_mut().zip(left_values).enumerate() {
            if index != left && index != right {
                *value = updated;
            }
        }
        for (index, (value, updated)) in matrix[right].iter_mut().zip(right_values).enumerate() {
            if index != left && index != right {
                *value = updated;
            }
        }
        for row in &mut eigenvectors {
            let left_value = row[left];
            let right_value = row[right];
            row[left] = cosine * left_value - sine * right_value;
            row[right] = sine * left_value + cosine * right_value;
        }
    }

    let mut minimizer = midpoint;
    for axis in 0..3 {
        let eigenvalue = matrix[axis][axis];
        if eigenvalue.abs() <= 0.1 {
            continue;
        }
        let projection = (eigenvectors[0][axis] * residual[0]
            + eigenvectors[1][axis] * residual[1]
            + eigenvectors[2][axis] * residual[2])
            / eigenvalue;
        for coordinate in 0..3 {
            minimizer[coordinate] += eigenvectors[coordinate][axis] * projection;
        }
    }
    (
        DVec3::from_array(minimizer.map(f64::from)),
        DVec3::from_array(midpoint.map(f64::from)),
    )
}
fn morton_cell(mut code: usize, depth: u32) -> [usize; 3] {
    let mut coordinates = [0; 3];
    for bit in 0..depth {
        let child = code & 7;
        code >>= 3;
        coordinates[0] |= ((child >> 2) & 1) << bit;
        coordinates[1] |= ((child >> 1) & 1) << bit;
        coordinates[2] |= (child & 1) << bit;
    }
    coordinates
}
