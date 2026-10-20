// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/subdiv_mesh.cc, SPDX-FileCopyrightText: 2018 Blender Authors, GPL-2.0-or-later.
//   source/blender/blenkernel/intern/subdiv_foreach.cc, SPDX-FileCopyrightText: 2018 Blender Authors, GPL-2.0-or-later.

//! Loop normals of a subdivided mesh: each output loop mixes its base face's corner normals at the
//! loop's own ptex sample. Blender's fan spaces also depend on the order of the loops, so this module
//! reproduces the order in which `subdiv_foreach` emits them.

use glam::{DVec3, Vec3};

use super::subdivision_limit::Sample;
use crate::geom::clnor::normalize;

/// The normal of every output loop, in output face order (`subdiv_interpolate_corner_data` with
/// `mix_normals`). `normals` holds the base corner normals in base face order, `face_sizes` the
/// corner count of each base face, and `loops[f]` the four ptex samples of output face `f`.
///
/// A quad base face mixes its corners with the bilinear weights of its sample. An n-gon's ptex face
/// `k` mixes four stored values: its corner `k`, the normalised mix of corners `k` and `k + 1`, the
/// normalised mean of all corners, and the normalised mix of corners `k - 1` and `k`.
pub(super) fn loop_normals(
    normals: &[DVec3],
    face_sizes: &[usize],
    loops: &[Vec<Sample>],
) -> Vec<DVec3> {
    let normals: Vec<Vec3> = normals.iter().map(|normal| normal.as_vec3()).collect();
    let mut starts = Vec::with_capacity(face_sizes.len());
    let mut centers = Vec::with_capacity(face_sizes.len());
    let mut start = 0;
    for &size in face_sizes {
        let face = &normals[start..start + size];
        starts.push(start);
        centers.push(if size == 4 {
            Vec3::ZERO
        } else {
            // Blender weights every corner by `1 / size` before it normalizes the sum.
            mix_equal(face, 1.0 / size as f32)
        });
        start += size;
    }
    loops
        .iter()
        .flatten()
        .map(|sample| {
            let face = sample.face;
            let corners = &normals[starts[face]..starts[face] + face_sizes[face]];
            let (u, v) = (sample.u as f32, sample.v as f32);
            let weights = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
            let stored = if face_sizes[face] == 4 {
                [corners[0], corners[1], corners[2], corners[3]]
            } else {
                let count = corners.len();
                let k = sample.ptex;
                let previous = corners[(k + count - 1) % count];
                let current = corners[k];
                let next = corners[(k + 1) % count];
                [
                    current,
                    midpoint(current, next),
                    centers[face],
                    midpoint(previous, current),
                ]
            };
            mix(&stored, &weights).as_dvec3()
        })
        .collect()
}

/// `mix_normals`: the weighted sum of the vectors, normalized.
fn mix(vectors: &[Vec3], weights: &[f32]) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for (vector, weight) in vectors.iter().zip(weights) {
        sum += *vector * *weight;
    }
    normalize(sum)
}

/// `mix_normals` with the same weight for every vector.
fn mix_equal(vectors: &[Vec3], weight: f32) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for &vector in vectors {
        sum += vector * weight;
    }
    normalize(sum)
}

/// `mix_normals` with factor one half: `math::interpolate`, then normalized.
fn midpoint(a: Vec3, b: Vec3) -> Vec3 {
    normalize(a * (1.0 - 0.5) + b * 0.5)
}

/// Blender's corner numbering of the subdivided mesh, one number per output corner in output face
/// order. Output face `f` gets the block `4 * E..4 * E + 4`, where `E` is the emission index of the
/// face: the faces of each base face form one block, in base face order. `loops[f]` holds the four
/// ptex samples of output face `f`, `base_sizes` the corner count of each base face, `loop_starts`
/// the corner holding each base face's first loop, and `levels` is at least one. The samples stay in
/// canonical geometry; each is moved into its base face's loop frame before its cell is located.
pub(super) fn corner_numbering(
    loops: &[Vec<Sample>],
    base_sizes: &[usize],
    loop_starts: &[usize],
    levels: u32,
) -> Vec<usize> {
    let grid = 1_i64 << levels;
    let ptex_grid = grid / 2;
    let block = |size: usize| {
        if size == 4 {
            grid * grid
        } else {
            size as i64 * ptex_grid * ptex_grid
        }
    };
    let mut starts = Vec::with_capacity(base_sizes.len());
    let mut total = 0;
    for &size in base_sizes {
        starts.push(total);
        total += block(size);
    }
    let mut numbering = Vec::with_capacity(loops.len() * 4);
    for face_loops in loops {
        let frame: [Sample; 4] = std::array::from_fn(|k| {
            let sample = face_loops[k];
            sample.in_loop_frame(loop_starts[sample.face], base_sizes[sample.face])
        });
        let first = frame[0];
        let size = base_sizes[first.face];
        let cells = if size == 4 { grid } else { ptex_grid };
        let low_u = frame
            .iter()
            .map(|sample| sample.u)
            .fold(f64::INFINITY, f64::min);
        let low_v = frame
            .iter()
            .map(|sample| sample.v)
            .fold(f64::INFINITY, f64::min);
        let x = (low_u * cells as f64).round() as i64;
        let y = (low_v * cells as f64).round() as i64;
        let emitted = if size == 4 {
            quad_cell_index(x, y, grid)
        } else {
            ngon_cell_index(x, y, first.ptex as i64, size as i64, ptex_grid)
        };
        let face = starts[first.face] + emitted;
        for sample in &frame {
            let local_u = ((sample.u - low_u) * cells as f64).round() as i64;
            let local_v = ((sample.v - low_v) * cells as f64).round() as i64;
            let corner = match (local_u, local_v) {
                (0, 0) => 0,
                (1, 0) => 1,
                (1, 1) => 2,
                (0, 1) => 3,
                _ => unreachable!("a Subsurf face loop is one of four grid corners"),
            };
            numbering.push(4 * face as usize + corner);
        }
    }
    numbering
}

/// Emission index of cell `(x, y)` in a quad's `n x n` grid (`subdiv_foreach_loops_regular`): the
/// inner cells row by row, then the ring along each side from corner 0 to corner 3.
fn quad_cell_index(x: i64, y: i64, n: i64) -> i64 {
    let inner = n - 2;
    if (1..=inner).contains(&x) && (1..=inner).contains(&y) {
        return (y - 1) * inner + (x - 1);
    }
    let ring = n - 1;
    let base = inner * inner;
    if y == 0 {
        base + x
    } else if x == n - 1 {
        base + ring + y
    } else if y == n - 1 {
        base + 2 * ring + (n - 1 - x)
    } else {
        base + 3 * ring + (n - 1 - y)
    }
}

/// Emission index of cell `(x, y)` in ptex face `ptex` of an n-gon with `size` corners and `g x g`
/// cells per ptex face (`subdiv_foreach_loops_special`). Every corner emits its inner rows, then
/// every corner emits the row shared with its next ptex face, the center cells, and the two edge
/// strips.
fn ngon_cell_index(x: i64, y: i64, ptex: i64, size: i64, g: i64) -> i64 {
    let inner_per_ptex = (g - 1) * (g - 2).max(0);
    let connection_base = size * inner_per_ptex;
    if (1..=g - 1).contains(&x) && (1..=g - 2).contains(&y) {
        return ptex * inner_per_ptex + (y - 1) * (g - 1) + (x - 1);
    }
    let connections_per_ptex = (g - 2).max(0);
    if y == g - 1 && (1..=g - 2).contains(&x) {
        let owner = (ptex + size - 1) % size;
        return connection_base + owner * connections_per_ptex + (x - 1);
    }
    let center_base = connection_base + size * connections_per_ptex;
    if g >= 2 && x == g - 1 && y == g - 1 {
        return center_base + ptex;
    }
    // `subdiv_foreach_loops_special` emits the per-corner center loops only when
    // `ptex_face_resolution >= 3`, that is `g >= 2`; a one-cell ptex face has none.
    let centers = if g >= 2 { size } else { 0 };
    let strip_base = center_base + centers;
    let per_ptex = 2 * g - 1;
    strip_base + ptex * per_ptex + if y == 0 { x } else { g + (y - 1) }
}
