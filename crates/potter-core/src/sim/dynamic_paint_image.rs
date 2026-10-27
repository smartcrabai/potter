// SPDX-License-Identifier: GPL-3.0-or-later
//
// Blender-derived portions: source/blender/blenkernel/intern/dynamicpaint.cc,
// SPDX-FileCopyrightText: 2023 Blender Authors, GPL-2.0-or-later.

//! Rasterizes a mesh's corner UVs to Blender Dynamic Paint IMAGE surface points.

use std::collections::HashMap;

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Mesh, uv_layers},
};

const MAX_IMAGE_TEXELS: usize = 1_000_000;
const NO_TRIANGLE: usize = usize::MAX;
const NO_PIXEL: usize = usize::MAX;

// Blender dynamicpaint.cc: neighStraightX/neighStraightY for UV-margin filling.
const STRAIGHT_X: [isize; 8] = [1, 0, -1, 0, 1, -1, -1, 1];
const STRAIGHT_Y: [isize; 8] = [0, 1, 0, -1, 1, 1, -1, -1];
const JITTER: [[f32; 2]; 5] = [
    [0.0, 0.0],
    [-0.2, -0.4],
    [0.2, 0.4],
    [0.4, -0.2],
    [-0.4, 0.3],
];

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ImageSurface {
    pub(crate) resolution: usize,
    pub(crate) samples_per_pixel: usize,
    pub(crate) pixels: Vec<ImagePixel>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ImagePixel {
    /// Blender's bottom-up flat texel index (`x + resolution * y`).
    pub(crate) pixel_index: usize,
    pub(crate) vertex_indices: [usize; 3],
    pub(crate) barycentric_weights: [[f32; 3]; 5],
}

#[derive(Clone, Copy)]
struct Triangle {
    vertex_indices: [usize; 3],
    uv: [[f32; 2]; 3],
}

pub(crate) fn build_surface(
    mesh: &Mesh,
    uv_layer: Option<&str>,
    resolution: usize,
    antialias: bool,
) -> Result<ImageSurface> {
    if resolution == 0 {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "dynamic paint image resolution must be nonzero",
        ));
    }
    let texel_count = resolution.checked_mul(resolution).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "dynamic paint image texel count exceeds the resource limit",
        )
    })?;
    if texel_count > MAX_IMAGE_TEXELS {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "dynamic paint image exceeds the one-million texel limit",
        ));
    }

    let store = uv_layers::read(mesh);
    let layer_index = uv_layer
        .and_then(|name| store.index_of(name))
        .or_else(|| {
            store
                .active
                .as_deref()
                .and_then(|name| store.index_of(name))
        })
        .or_else(|| store.layers.first().map(|_| 0))
        .ok_or_else(|| scene_invalid("dynamic paint image mesh has no UV layer"))?;
    let layer = &store.layers[layer_index];

    let vertex_indices: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut triangles = Vec::new();
    let mut triangle_error = None;
    mesh.for_each_triangle_with_face(|face_index, ids| {
        if triangle_error.is_some() {
            return;
        }
        let Some(face) = mesh.faces.get(face_index) else {
            triangle_error = Some(scene_invalid(
                "dynamic paint triangle references a missing mesh face",
            ));
            return;
        };
        let Some(Some(face_uv)) = layer.faces.get(face_index) else {
            triangle_error = Some(scene_invalid(
                "dynamic paint UV layer has no data for a triangle face",
            ));
            return;
        };
        if face_uv.len() != face.vertices.len() {
            triangle_error = Some(scene_invalid(
                "dynamic paint UV corner count does not match its mesh face",
            ));
            return;
        }

        let mut triangle_vertices = [0; 3];
        let mut triangle_uv = [[0.0; 2]; 3];
        for corner in 0..3 {
            let id = ids[corner];
            let Some(vertex_index) = vertex_indices.get(&id).copied() else {
                triangle_error = Some(scene_invalid(
                    "dynamic paint triangle references a missing mesh vertex",
                ));
                return;
            };
            let Some(face_corner) = face.vertices.iter().position(|&face_id| face_id == id) else {
                triangle_error = Some(scene_invalid(
                    "dynamic paint triangle vertex is not a corner of its source face",
                ));
                return;
            };
            let uv = face_uv[face_corner];
            // Potter stores UV V down; Blender's IMAGE surface grid is V up.
            let blender_uv = [uv[0] as f32, (1.0 - uv[1]) as f32];
            if !uv[0].is_finite()
                || !uv[1].is_finite()
                || !blender_uv[0].is_finite()
                || !blender_uv[1].is_finite()
            {
                triangle_error = Some(scene_invalid(
                    "dynamic paint UV coordinates must be finite single-precision values",
                ));
                return;
            }
            triangle_vertices[corner] = vertex_index;
            triangle_uv[corner] = blender_uv;
        }
        triangles.push(Triangle {
            vertex_indices: triangle_vertices,
            uv: triangle_uv,
        });
    })
    .map_err(|error| scene_invalid(&format!("invalid dynamic paint mesh: {error}")))?;
    if let Some(error) = triangle_error {
        return Err(error);
    }

    let mut triangle_by_pixel = vec![NO_TRIANGLE; texel_count];
    let mut border_neighbor = vec![NO_PIXEL; texel_count];
    let resolution_f32 = resolution as f32;
    let mut triangle_bounds = Vec::with_capacity(triangles.len());
    for triangle in &triangles {
        let mut min = triangle.uv[0];
        let mut max = triangle.uv[0];
        for uv in &triangle.uv[1..] {
            min[0] = min[0].min(uv[0]);
            min[1] = min[1].min(uv[1]);
            max[0] = max[0].max(uv[0]);
            max[1] = max[1].max(uv[1]);
        }
        triangle_bounds.push((min, max));
    }

    // Blender's direct callback tests center, lower-left, lower-right, upper-left, upper-right,
    // and accepts the first triangle in corner-triangle order containing any of those samples.
    for y in 0..resolution {
        for x in 0..resolution {
            let texel = x + resolution * y;
            let x_f = x as f32;
            let y_f = y as f32;
            let test_points = [
                [(x_f + 0.5) / resolution_f32, (y_f + 0.5) / resolution_f32],
                [x_f / resolution_f32, y_f / resolution_f32],
                [(x_f + 1.0) / resolution_f32, y_f / resolution_f32],
                [x_f / resolution_f32, (y_f + 1.0) / resolution_f32],
                [(x_f + 1.0) / resolution_f32, (y_f + 1.0) / resolution_f32],
            ];
            'samples: for point in test_points {
                for (triangle_index, triangle) in triangles.iter().enumerate() {
                    let (min, max) = triangle_bounds[triangle_index];
                    if min[0] > point[0]
                        || min[1] > point[1]
                        || max[0] < point[0]
                        || max[1] < point[1]
                        || !point_in_triangle(point, triangle.uv)
                    {
                        continue;
                    }
                    triangle_by_pixel[texel] = triangle_index;
                    break 'samples;
                }
            }
        }
    }

    // Fill one-pixel UV margins from direct pixels only, in Blender's straight-first order.
    for y in 0..resolution {
        for x in 0..resolution {
            let texel = x + resolution * y;
            if triangle_by_pixel[texel] != NO_TRIANGLE {
                continue;
            }
            for direction in 0..8 {
                let nx = STRAIGHT_X[direction];
                let ny = STRAIGHT_Y[direction];
                let candidate_x = x as isize + nx;
                let candidate_y = y as isize + ny;
                if candidate_x < 0
                    || candidate_x >= resolution as isize
                    || candidate_y < 0
                    || candidate_y >= resolution as isize
                {
                    continue;
                }
                let candidate = candidate_x as usize + resolution * candidate_y as usize;
                if border_neighbor[candidate] == NO_PIXEL
                    && triangle_by_pixel[candidate] != NO_TRIANGLE
                {
                    border_neighbor[texel] = candidate;
                    triangle_by_pixel[texel] = triangle_by_pixel[candidate];
                    break;
                }
            }
        }
    }

    let active_count = triangle_by_pixel
        .iter()
        .filter(|&&triangle_index| triangle_index != NO_TRIANGLE)
        .count();
    if active_count == 0 {
        return Err(scene_invalid(
            "dynamic paint image UV map covers no texels at the requested resolution",
        ));
    }
    let mut pixels = Vec::with_capacity(active_count);
    let samples_per_pixel = if antialias { JITTER.len() } else { 1 };
    for (pixel_index, &triangle_index) in triangle_by_pixel.iter().enumerate() {
        if triangle_index == NO_TRIANGLE {
            continue;
        }
        let x = pixel_index % resolution;
        let y = pixel_index / resolution;
        let center = [
            (x as f32 + 0.5) / resolution_f32,
            (y as f32 + 0.5) / resolution_f32,
        ];
        let triangle = triangles[triangle_index];
        let mut barycentric_weights = [[1.0 / 3.0; 3]; 5];
        for (sample_index, sample) in JITTER.iter().take(samples_per_pixel).enumerate() {
            let uv = [
                center[0] + sample[0] / resolution_f32,
                center[1] + sample[1] / resolution_f32,
            ];
            barycentric_weights[sample_index] = barycentric_weights_v2(triangle.uv, uv);
        }
        pixels.push(ImagePixel {
            pixel_index,
            vertex_indices: triangle.vertex_indices,
            barycentric_weights,
        });
    }

    Ok(ImageSurface {
        resolution,
        samples_per_pixel,
        pixels,
    })
}

fn point_in_triangle(point: [f32; 2], triangle: [[f32; 2]; 3]) -> bool {
    let [first, second, third] = triangle;
    let side12 = line_point_side(first, second, point);
    let side23 = line_point_side(second, third, point);
    let side31 = line_point_side(third, first, point);
    (side12 >= 0.0 && side23 >= 0.0 && side31 >= 0.0)
        || (side12 <= 0.0 && side23 <= 0.0 && side31 <= 0.0)
}

fn line_point_side(first: [f32; 2], second: [f32; 2], point: [f32; 2]) -> f32 {
    (first[0] - point[0]) * (second[1] - point[1]) - (second[0] - point[0]) * (first[1] - point[1])
}

fn barycentric_weights_v2(triangle: [[f32; 2]; 3], point: [f32; 2]) -> [f32; 3] {
    let [first, second, third] = triangle;
    let mut weights = [
        cross_tri_v2(second, third, point),
        cross_tri_v2(third, first, point),
        cross_tri_v2(first, second, point),
    ];
    let total = weights[0] + weights[1] + weights[2];
    let inverse = 1.0 / total;
    for weight in &mut weights {
        *weight *= inverse;
    }
    if weights.iter().all(|weight| weight.is_finite()) {
        weights
    } else {
        [1.0 / 3.0; 3]
    }
}

fn cross_tri_v2(first: [f32; 2], second: [f32; 2], third: [f32; 2]) -> f32 {
    (first[0] - second[0]) * (second[1] - third[1])
        + (first[1] - second[1]) * (third[0] - second[0])
}

fn scene_invalid(message: &str) -> PotError {
    PotError::new(ErrorCode::SceneInvalid, message)
}
