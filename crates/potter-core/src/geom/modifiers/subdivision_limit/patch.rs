// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from OpenSubdiv (https://graphics.pixar.com/opensubdiv) far/patchBasis.cpp, far/patchBuilder.cpp (IsPatchRegular, getQuadRegularPatchPoints, GetRegularPatchBoundaryMask), far/catmarkPatchBuilder.cpp, Apache-2.0.

//! Leaf patches: the uniform cubic B-spline for regular quads and the Gregory endcap otherwise. A
//! patch is a sparse combination of level vertices, so each evaluation is a fixed weighted sum of
//! the leaf's control points.

use std::f64::consts::PI;

use super::gregory::{corner_span, fan, is_singular_edge};
use super::level::Level;

/// A sparse affine combination of level vertices: `(vertex, weight)`.
pub(super) type Row = Vec<(usize, f64)>;

/// Weighted sum of rows.
pub(super) fn combine(terms: &[(f64, &Row)]) -> Row {
    let mut out: Row = Vec::new();
    for (weight, row) in terms {
        for &(vertex, w) in *row {
            out.push((vertex, w * weight));
        }
    }
    out
}

/// Cubic B-spline basis of a normalized parameter (`OpenSubdiv` `EvalBasisBSpline`).
fn bspline_basis(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        (1.0 - 3.0 * (t - t2) - t3) / 6.0,
        (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
        (1.0 + 3.0 * (t + t2 - t3)) / 6.0,
        t3 / 6.0,
    ]
}

/// Cubic Bezier basis (`OpenSubdiv` `evalBezierCurve`).
fn bezier_basis(t: f64) -> [f64; 4] {
    let it = 1.0 - t;
    [it * it * it, 3.0 * t * it * it, 3.0 * t * t * it, t * t * t]
}

/// Control-point indices of the 4x4 B-spline grid touched by each corner's ring
/// (`OpenSubdiv` `getQuadRegularPatchPoints`); `k = 4 * row + col` with row along `t`, col along `s`.
const CORNER_POINTS: [[usize; 4]; 4] =
    [[5, 4, 0, 1], [6, 2, 3, 7], [10, 11, 15, 14], [9, 13, 12, 8]];

/// Boundary mask of a leaf: bit `i` is set when its edge from corner `i` to `i + 1` is singular
/// (boundary, non-manifold, or infinitely sharp).
pub(super) fn boundary_mask(level: &Level, face: usize) -> u8 {
    (0..4).fold(0, |mask, i| {
        mask | (u8::from(is_singular_edge(level, level.face_edges[face][i])) << i)
    })
}

/// Control points of a regular quad: the 4x4 grid, with the outer points taken from the opposite
/// face around each corner (or the corner itself where a boundary removes them).
#[expect(
    clippy::expect_used,
    reason = "a boundary-adjacent interior edge always has a neighbouring face on the manifold side"
)]
pub(super) fn regular_grid(level: &Level, face: usize, mask: u8) -> [Row; 16] {
    let corners = level.faces[face].clone();
    let boundary_point: Row = vec![(corners[0], 1.0)];
    let mut grid: [Row; 16] = std::array::from_fn(|_| Vec::new());
    for c in 0..4 {
        let vertex = corners[c];
        let out_boundary = mask & (1 << c) != 0;
        let in_boundary = mask & (1 << ((c + 3) % 4)) != 0;
        let mut outer: [Row; 3] = [
            boundary_point.clone(),
            boundary_point.clone(),
            boundary_point.clone(),
        ];
        if !out_boundary && !in_boundary {
            let ring = fan(level, face, vertex);
            let opposite = ring[2];
            let vin = level.corner_of(opposite, vertex);
            let verts = &level.faces[opposite];
            outer = [
                vec![(verts[(vin + 1) % 4], 1.0)],
                vec![(verts[(vin + 2) % 4], 1.0)],
                vec![(verts[(vin + 3) % 4], 1.0)],
            ];
        } else if out_boundary && in_boundary {
            // Both incident edges are boundary: the corner has no exterior points.
        } else if out_boundary {
            let next = level
                .other_face(level.face_edges[face][(c + 3) % 4], face)
                .expect("a boundary-adjacent interior edge has a neighbour");
            let vin = level.corner_of(next, vertex);
            outer[0] = vec![(level.faces[next][(vin + 3) % 4], 1.0)];
        } else {
            let prev = level
                .other_face(level.face_edges[face][c], face)
                .expect("a boundary-adjacent interior edge has a neighbour");
            let vin = level.corner_of(prev, vertex);
            outer[2] = vec![(level.faces[prev][(vin + 1) % 4], 1.0)];
        }
        let slots = CORNER_POINTS[c];
        grid[slots[0]] = vec![(vertex, 1.0)];
        grid[slots[1]].clone_from(&outer[0]);
        grid[slots[2]].clone_from(&outer[1]);
        grid[slots[3]].clone_from(&outer[2]);
    }
    grid
}

/// B-spline weights at `(s, t)` on the 16-point grid, with the boundary mask applied
/// (`OpenSubdiv` `EvalBasisBSpline` and `adjustBSplineBoundaryWeights`).
pub(super) fn bspline_weights(s: f64, t: f64, mask: u8) -> [f64; 16] {
    let bs = bspline_basis(s);
    let bt = bspline_basis(t);
    let mut w = [0.0; 16];
    for row in 0..4 {
        for col in 0..4 {
            w[4 * row + col] = bs[col] * bt[row];
        }
    }
    if mask & 1 != 0 {
        for i in 0..4 {
            w[i + 8] -= w[i];
            w[i + 4] += 2.0 * w[i];
            w[i] = 0.0;
        }
    }
    if mask & 2 != 0 {
        for i in [0, 4, 8, 12] {
            w[i + 1] -= w[i + 3];
            w[i + 2] += 2.0 * w[i + 3];
            w[i + 3] = 0.0;
        }
    }
    if mask & 4 != 0 {
        for i in 0..4 {
            w[i + 4] -= w[i + 12];
            w[i + 8] += 2.0 * w[i + 12];
            w[i + 12] = 0.0;
        }
    }
    if mask & 8 != 0 {
        for i in [0, 4, 8, 12] {
            w[i + 2] -= w[i];
            w[i + 1] += 2.0 * w[i];
            w[i] = 0.0;
        }
    }
    w
}

/// Gregory basis of the 20 control points of a quad with an irregular corner (`OpenSubdiv`
/// `EvalBasisGregory`).
pub(super) fn gregory_weights(s: f64, t: f64) -> [f64; 20] {
    let bs = bezier_basis(s);
    let bt = bezier_basis(t);
    let (sc, tc) = (1.0 - s, 1.0 - t);
    let guard = |sum: f64| if sum > 0.0 { 1.0 / sum } else { 1.0 };
    let df0 = guard(s + t);
    let df1 = guard(sc + t);
    let df2 = guard(sc + tc);
    let df3 = guard(s + tc);
    let g0 = s * df0;
    let g2 = t * df1;
    let g4 = sc * df2;
    let g6 = tc * df3;
    let g = [g0, 1.0 - g0, g2, 1.0 - g2, g4, 1.0 - g4, g6, 1.0 - g6];
    let boundary_index = [0, 1, 7, 5, 2, 6, 16, 12, 15, 17, 11, 10];
    let boundary_s = [0, 1, 2, 3, 0, 3, 0, 3, 0, 1, 2, 3];
    let boundary_t = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3];
    let interior_index = [3, 4, 8, 9, 13, 14, 18, 19];
    let interior_s = [1, 1, 2, 2, 2, 2, 1, 1];
    let interior_t = [1, 1, 1, 1, 2, 2, 2, 2];
    let mut w = [0.0; 20];
    for i in 0..12 {
        w[boundary_index[i]] = bs[boundary_s[i]] * bt[boundary_t[i]];
    }
    for i in 0..8 {
        w[interior_index[i]] = bs[interior_s[i]] * bt[interior_t[i]] * g[i];
    }
    w
}

/// Catmull-Clark tangent coefficient for valence `n` (`OpenSubdiv` `CatmarkLimits::computeCoefficient`).
pub(super) fn tangent_coefficient(n: usize) -> f64 {
    const TABLE: [f64; 30] = [
        0.0,
        0.0,
        0.0,
        8.128_157_290_637_231e-1,
        0.5,
        3.636_440_632_914_28e-1,
        2.875_137_970_607_708_5e-01,
        2.386_878_668_585_167_8e-01,
        2.045_436_419_075_609_7e-01,
        1.792_290_395_806_116e-1,
        1.596_573_707_998_625_3e-01,
        1.440_423_344_301_130_2e-01,
        1.312_756_841_588_301_7e-01,
        1.206_317_221_267_584e-1,
        1.116_143_750_667_693e-1,
        1.038_724_551_611_427_4e-01,
        9.715_001_909_072_483e-2,
        9.125_591_750_595_065e-2,
        8.604_437_851_160_267e-2,
        8.140_221_133_679_841e-2,
        7.724_012_951_618_407e-2,
        7.348_671_975_199_703e-2,
        7.008_415_747_979_799e-2,
        6.698_510_403_072_544e-2,
        6.415_042_056_981_007e-2,
        6.154_745_763_863_727e-2,
        5.914_875_744_723_399e-2,
        5.693_105_681_877_696e-2,
        5.487_451_227_925_642e-2,
        5.296_209_143_379_613_4e-02,
    ];
    if n < 30 {
        TABLE[n]
    } else {
        let inv = 1.0 / n as f64;
        let cos_t = (2.0 * PI * inv).cos();
        let divisor = (cos_t + 5.0) + ((cos_t + 9.0) * (cos_t + 1.0)).sqrt();
        16.0 * inv / divisor
    }
}

/// Whether the leaf is evaluated by the B-spline (`OpenSubdiv` `IsPatchRegular`). `depth` is the
/// leaf's level; `has_irregular_faces` is whether the base mesh has non-quad faces.
pub(super) fn is_regular_leaf(
    level: &Level,
    face: usize,
    depth: usize,
    has_irregular_faces: bool,
) -> bool {
    let corners = level.faces[face].clone();
    let tags: Vec<_> = corners.iter().map(|&v| level.vtags[v]).collect();
    let combined = super::level::VTag::combine(&tags);
    if !combined.inf_sharp && !combined.inf_sharp_edges {
        return !combined.xordinary;
    }
    let irregular = combined.non_manifold || combined.xordinary || combined.inf_irregular;
    if !irregular {
        return true;
    }
    let needs_extra_level = combined.xordinary && has_irregular_faces;
    if depth > usize::from(needs_extra_level) {
        let requires_inspection = combined.non_manifold
            || (combined.xordinary && combined.boundary)
            || (combined.inf_irregular && combined.inf_sharp_edges);
        if !requires_inspection {
            return !combined.inf_irregular;
        }
    }
    for (i, tag) in tags.iter().enumerate() {
        if !(tag.non_manifold || tag.xordinary || tag.inf_irregular) {
            continue;
        }
        if tag.non_manifold {
            // Non-manifold corners are only regular when their span is a single face with an
            // infinitely sharp boundary, which the sharp-corner path reproduces.
            return false;
        }
        if tag.xordinary {
            if !tag.inf_sharp_edges {
                return false;
            }
            if tag.boundary && !tag.inf_sharp {
                let previous = (i + 3) % 4;
                let out_boundary = is_singular_edge(level, level.face_edges[face][i]);
                let in_boundary = is_singular_edge(level, level.face_edges[face][previous]);
                if out_boundary && in_boundary {
                    continue;
                }
            }
        }
        if tag.inf_irregular {
            if !tag.inf_sharp_edges {
                return false;
            }
            if tag.inf_sharp_crease && tag.boundary {
                return false;
            }
            let span = corner_span(level, face, corners[i]);
            let span_faces = span.faces.len();
            let expected = if tag.inf_sharp_crease { 2 } else { 1 };
            if span_faces != expected {
                return false;
            }
        }
    }
    true
}
