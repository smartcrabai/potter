// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   intern/dualcon/intern/dualcon_c_api.cpp (DualConInputReader, dualcon), SPDX-FileCopyrightText:
//   2011-2023 Blender Authors, GPL-2.0-or-later.
//   source/blender/modifiers/intern/MOD_remesh.cc, SPDX-FileCopyrightText: 2011 by Nicholas Bishop,
//   GPL-2.0-or-later.

//! Dual-contouring Remesh: a port of Blender's `dualcon()` octree pipeline for BLOCKS, SMOOTH, and
//! SHARP modes. Open surfaces, non-manifold edges, and disconnected components follow the same
//! path as closed meshes, including the in-octree flood fill.

mod octree;
mod output;
mod patch;
mod projection;
mod qef;
mod signs;

/// Error for a broken octree invariant. Blender cannot reach these states; Potter reports them
/// instead of reading through a missing node.
pub(super) fn invariant(message: &str) -> PotError {
    PotError::invalid_operation(format!(
        "dual contour internal invariant violated: {message}"
    ))
}
mod tables;
mod walk;

use std::collections::HashMap;

use glam::{DVec3, Vec3};

use crate::{
    error::{PotError, Result},
    geom::Mesh,
};

use octree::{MAX_DUALCON_WORK_ITEMS, Mode, Octree, allocation_limit};

/// Parameters of one `DualCon` run (`dualcon()` arguments).
pub(super) struct DualConSettings<'a> {
    pub(super) mode: &'a str,
    pub(super) depth: u32,
    pub(super) scale: f32,
    pub(super) sharpness: f32,
    pub(super) threshold: f32,
    pub(super) flood_fill: bool,
}

pub(super) fn remesh(mesh: &Mesh, settings: &DualConSettings<'_>) -> Result<Mesh> {
    let mode = match settings.mode {
        "BLOCKS" => Mode::Centroid,
        "SMOOTH" => Mode::MassPoint,
        "SHARP" => Mode::Sharp,
        _ => return Err(PotError::invalid_operation("unsupported dual contour mode")),
    };
    let Some(bounds) = mesh.bounds() else {
        return Ok(Mesh::new());
    };
    let (origin, range) =
        reader_bounds(bounds.min.to_array(), bounds.max.to_array(), settings.scale);

    let triangles = blender_corner_triangles(mesh)?;
    let input_work_items = mesh
        .vertices
        .len()
        .checked_add(triangles.capacity())
        .ok_or_else(allocation_limit)?;

    let depth = i32::try_from(settings.depth)
        .map_err(|_| PotError::invalid_operation("dual contour depth exceeds integer range"))?;
    let mut octree = Octree::new(
        depth,
        origin,
        range,
        settings.threshold,
        settings.sharpness,
        mode,
        input_work_items,
    )?;
    octree.scan_convert(&triangles)?;
    // Blender calls `trace()` twice; the second pass patches rings left by the first.
    octree.trace_root()?;
    octree.trace_root()?;
    octree.build_signs();
    if settings.flood_fill {
        octree.flood_fill()?;
        octree.build_signs();
    }
    octree.write_out()?;

    output_mesh(&octree)
}

/// `DualConInputReader::reset` and `getBoundingBox`: the cubic grid that contains the input.
#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's f32 evaluation order in DualConInputReader"
)]
fn reader_bounds(min: [f64; 3], max: [f64; 3], scale: f32) -> ([f32; 3], f32) {
    let mut lower: [f32; 3] = min.map(|value| value as f32);
    let mut upper: [f32; 3] = max.map(|value| value as f32);

    let mut size = 0.0_f32;
    for axis in 0..3 {
        let extent = upper[axis] - lower[axis];
        size = size.max(extent);
    }

    // Blender reads `max[i]` after `min[i]` was replaced, so the order of these two statements is
    // part of the contract.
    for axis in 0..3 {
        lower[axis] = (upper[axis] + lower[axis]) / 2.0 - size / 2.0;
        upper[axis] = (upper[axis] + lower[axis]) / 2.0 + size / 2.0;
    }
    let inverse_scale = 1.0_f32 / scale;
    for value in &mut lower {
        *value -= size * (inverse_scale - 1.0) / 2.0;
    }
    (lower, size * inverse_scale)
}

fn output_mesh(octree: &Octree) -> Result<Mesh> {
    if octree.vertices.is_empty() && octree.quads.is_empty() {
        return Ok(Mesh::new());
    }
    let vertex_count = octree.vertices.len();
    let face_count = octree.quads.len();
    let corner_count = face_count.checked_mul(4).ok_or_else(allocation_limit)?;
    octree.check_work_items(
        vertex_count
            .checked_add(face_count)
            .and_then(|count| count.checked_add(corner_count))
            .ok_or_else(allocation_limit)?,
    )?;
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(vertex_count)
        .map_err(|_| allocation_limit())?;
    positions.extend(
        octree
            .vertices
            .iter()
            .map(|point| DVec3::from_array(point.map(f64::from))),
    );
    let mut polygons = Vec::new();
    polygons
        .try_reserve_exact(face_count)
        .map_err(|_| allocation_limit())?;
    for quad in &octree.quads {
        let mut polygon = Vec::new();
        polygon
            .try_reserve_exact(4)
            .map_err(|_| allocation_limit())?;
        for index in quad {
            polygon.push(
                usize::try_from(*index)
                    .ok()
                    .filter(|index| *index < vertex_count)
                    .ok_or_else(|| {
                        PotError::invalid_operation(
                            "dual contour quad references a vertex outside the output",
                        )
                    })?,
            );
        }
        polygons.push(polygon);
    }
    Mesh::from_positions_and_faces(positions, polygons).map_err(|error| {
        PotError::invalid_argument(format!("dual contour mesh construction failed: {error}"))
    })
}

fn blender_corner_triangles(mesh: &Mesh) -> Result<Vec<[[f32; 3]; 3]>> {
    let triangle_count = mesh.faces.iter().try_fold(0_usize, |count, face| {
        count.checked_add(face.vertices.len().saturating_sub(2))
    });
    let triangle_count = triangle_count.ok_or_else(allocation_limit)?;
    let input_items = mesh
        .vertices
        .len()
        .checked_add(triangle_count)
        .ok_or_else(allocation_limit)?;
    if input_items >= MAX_DUALCON_WORK_ITEMS {
        return Err(allocation_limit());
    }

    let mut vertex_indices = HashMap::new();
    vertex_indices
        .try_reserve(mesh.vertices.len())
        .map_err(|_| allocation_limit())?;
    for (index, vertex) in mesh.vertices.iter().enumerate() {
        vertex_indices.insert(vertex.id, index);
    }

    let mut triangles = Vec::new();
    triangles
        .try_reserve_exact(triangle_count)
        .map_err(|_| allocation_limit())?;
    let position = |index: usize| {
        mesh.vertices[index]
            .co
            .to_array()
            .map(|coordinate| coordinate as f32)
    };
    for face in &mesh.faces {
        let mut vertices = Vec::new();
        vertices
            .try_reserve_exact(face.vertices.len())
            .map_err(|_| allocation_limit())?;
        for vertex in &face.vertices {
            vertices.push(vertex_indices.get(vertex).copied().ok_or_else(|| {
                PotError::invalid_operation("dual contour face references an unknown vertex")
            })?);
        }
        if vertices.len() == 4 {
            let points = std::array::from_fn(|corner| Vec3::from_array(position(vertices[corner])));
            if blender_quad_uses_alternate_diagonal(points) {
                triangles.push([
                    points[0].to_array(),
                    points[1].to_array(),
                    points[3].to_array(),
                ]);
                triangles.push([
                    points[1].to_array(),
                    points[2].to_array(),
                    points[3].to_array(),
                ]);
            } else {
                triangles.push([
                    points[0].to_array(),
                    points[1].to_array(),
                    points[2].to_array(),
                ]);
                triangles.push([
                    points[0].to_array(),
                    points[2].to_array(),
                    points[3].to_array(),
                ]);
            }
        } else {
            for corner in 1..vertices.len().saturating_sub(1) {
                triangles.push([
                    position(vertices[0]),
                    position(vertices[corner]),
                    position(vertices[corner + 1]),
                ]);
            }
        }
    }
    triangles.retain(|triangle| {
        triangle
            .iter()
            .all(|point| point.iter().all(|value| !value.is_nan()))
    });
    Ok(triangles)
}

fn blender_quad_uses_alternate_diagonal(points: [Vec3; 4]) -> bool {
    let first_normal = (points[1] - points[0]).cross(points[2] - points[0]);
    let second_normal = (points[3] - points[0]).cross(points[2] - points[0]);
    first_normal.dot(second_normal) > 0.0
}
