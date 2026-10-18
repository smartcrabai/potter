// SPDX-License-Identifier: GPL-3.0-or-later

//! Blender `Solidify` in `NON_MANIFOLD` mode, ported from `MOD_solidify_nonmanifold.cc`.
//!
//! Blender's pointer graph becomes indices: side `2 * face + reversed` is a `NewFaceRef`, an index
//! into `new_edges` is a `NewEdgeRef`, and `(vertex, index)` names an `EdgeGroup`. `EMPTY` is
//! Blender's `MOD_SOLIDIFY_EMPTY_TAG`. Positions and offsets are `f32`, as Blender's `float`.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use glam::DVec3;
use serde_json::{Map, Value, json};

use super::super::{Polygon, build_mesh, edge_pairs, id_indices, mesh_attribute_values, polygons};
use super::{BoundaryMode, Params, ThicknessMode};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Mesh, edge_key, provenance::Origin},
    model::Modifier,
};

const EMPTY: usize = usize::MAX;
const PI_F32: f32 = std::f32::consts::PI;
const PI_F64: f64 = std::f64::consts::PI;
/// `FLT_EPSILON`.
const EPS: f32 = f32::EPSILON;

type V3 = [f32; 3];

fn to_v3(point: DVec3) -> V3 {
    [point.x as f32, point.y as f32, point.z as f32]
}

fn to_dvec(point: V3) -> DVec3 {
    DVec3::new(
        f64::from(point[0]),
        f64::from(point[1]),
        f64::from(point[2]),
    )
}

fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul3(a: V3, factor: f32) -> V3 {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

fn negate3(a: V3) -> V3 {
    [-a[0], -a[1], -a[2]]
}

fn dot3(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn len3(a: V3) -> f32 {
    dot3(a, a).sqrt()
}

/// Blender `normalize_v3`: scales `v` to unit length and returns the old length. A vector whose
/// squared length is at most `1e-35` is left unchanged and reports zero.
fn normalize3(v: &mut V3) -> f32 {
    let mut length = dot3(*v, *v);
    if length > 1.0e-35 {
        length = length.sqrt();
        *v = mul3(*v, 1.0 / length);
    } else {
        length = 0.0;
    }
    length
}

/// Blender `project_v3_v3`: removes the component of `r` along unit `a` and returns the dot product.
fn project3(r: &mut V3, a: V3) -> f32 {
    let d = dot3(*r, a);
    *r = sub3(*r, mul3(a, d));
    d
}

/// Blender `min_ff` and `max_ff`, which keep the first argument on ties.
fn min_ff(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

fn max_ff(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}
#[expect(
    clippy::manual_midpoint,
    reason = "preserve Blender's sum-then-half float evaluation order"
)]
fn blender_midpoint(a: f32, b: f32) -> f32 {
    (a + b) * 0.5
}

/// `std::max(a, b)` as Blender's `float` code calls it.
fn max_std(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

/// Blender `clamp_f`.
fn clamp_f(value: f32, min: f32, max: f32) -> f32 {
    if value > max {
        max
    } else if value < min {
        min
    } else {
        value
    }
}

/// `clamp_nonzero`: the closest value with `abs(value) >= epsilon`.
fn clamp_nonzero(value: f32, epsilon: f32) -> f32 {
    if value < 0.0 {
        min_ff(value, -epsilon)
    } else {
        max_ff(value, epsilon)
    }
}

/// `angle_normalized_v3v3`: the angle between unit vectors, computed from their chord length.
fn angle_normalized(v1: V3, v2: V3) -> f32 {
    if dot3(v1, v2) < 0.0 {
        let vec = negate3(v2);
        return PI_F32 - 2.0 * (len3(sub3(v1, vec)) / 2.0).asin();
    }
    2.0 * (len3(sub3(v1, v2)) / 2.0).asin()
}

/// `angle_v3v3v3`: the angle at `b` between `a` and `c`.
fn angle_v3v3v3(a: V3, b: V3, c: V3) -> f32 {
    let mut v1 = sub3(a, b);
    let mut v2 = sub3(c, b);
    normalize3(&mut v1);
    normalize3(&mut v2);
    angle_normalized(v1, v2)
}

/// `angle_signed_on_axis_normalized_v3v3_v3`.
fn angle_signed(n: V3, ref_n: V3, axis: V3) -> f32 {
    let d = dot3(n, ref_n).clamp(-1.0, 1.0);
    let mut angle = d.acos();
    if dot3(cross3(n, ref_n), axis) >= 0.0 {
        angle = (2.0 * PI_F64 - f64::from(angle)) as f32;
    }
    angle
}

/// Blender `invert_m3` on a row-major matrix. Blender adds `5e-5` to the diagonal before inverting,
/// so the determinant is nonzero for every reachable input; a zero determinant yields zeros.
fn invert3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
        - m[0][1] * (m[1][0] * m[2][2] - m[2][0] * m[1][2])
        + m[0][2] * (m[1][0] * m[2][1] - m[2][0] * m[1][1]);
    if det == 0.0 {
        return [[0.0; 3]; 3];
    }
    let inv = 1.0 / det;
    [
        [
            (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * inv,
            -(m[0][1] * m[2][2] - m[2][1] * m[0][2]) * inv,
            (m[0][1] * m[1][2] - m[1][1] * m[0][2]) * inv,
        ],
        [
            -(m[1][0] * m[2][2] - m[2][0] * m[1][2]) * inv,
            (m[0][0] * m[2][2] - m[2][0] * m[0][2]) * inv,
            -(m[0][0] * m[1][2] - m[1][0] * m[0][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * inv,
            -(m[0][0] * m[2][1] - m[2][0] * m[0][1]) * inv,
            (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * inv,
        ],
    ]
}

/// `NewFaceRef`: one oriented side of an original face. Its id is `2 * face + reversed`.
struct Side {
    face: usize,
    reversed: bool,
    start: usize,
    size: usize,
    /// `link_edges`: the new edge at each corner of the side, or `EMPTY`.
    links: Vec<usize>,
}

/// `NewEdgeRef`: one new edge of an original edge, between two sides.
struct NewEdge {
    old: usize,
    /// Side ids. The second is `EMPTY` for an edge with one adjacent face.
    faces: [usize; 2],
    /// `(vertex, group index)` of the edge group at each end of the original edge.
    links: [Option<(usize, usize)>; 2],
    angle: f32,
    /// Output edge index. `0` marks an edge to be built, `EMPTY` one that is not built in this mode.
    output_edge: usize,
}

/// `OldEdgeFaceRef`: the faces around an original edge. Several edges can share one object after a
/// collapse, which `used` counts.
struct AdjFaces {
    faces: Vec<usize>,
    reversed: Vec<bool>,
    used: usize,
}

/// `OldVertEdgeRef`: the original edges at a vertex; the first `edges_len` entries are in use.
struct VertEdges {
    edges: Vec<usize>,
    edges_len: usize,
}

/// `EdgeGroup`: the new edges of one vertex's fan, consecutive around the vertex.
#[derive(Clone)]
struct EdgeGroup {
    valid: bool,
    edges: Vec<usize>,
    is_orig_closed: bool,
    is_even_split: bool,
    split: usize,
    is_singularity: bool,
    topo_group: usize,
    co: V3,
    new_vert: usize,
}

impl Default for EdgeGroup {
    fn default() -> Self {
        Self {
            valid: false,
            edges: Vec::new(),
            is_orig_closed: false,
            is_even_split: false,
            split: 0,
            is_singularity: false,
            topo_group: 0,
            co: [0.0; 3],
            new_vert: EMPTY,
        }
    }
}

/// Number of valid groups: the prefix before the first invalid entry, as Blender's loops stop there.
fn valid_len(groups: &[EdgeGroup]) -> usize {
    groups
        .iter()
        .position(|group| !group.valid)
        .unwrap_or(groups.len())
}

/// Blender's `memmove` of `count` groups from `src` to `dst`.
fn shift_groups(groups: &mut [EdgeGroup], dst: usize, src: usize, count: usize) {
    let moved: Vec<EdgeGroup> = groups[src..src + count].to_vec();
    for (offset, group) in moved.into_iter().enumerate() {
        groups[dst + offset] = group;
    }
}

fn link_group(
    groups: &[Option<Vec<EdgeGroup>>],
    link: Option<(usize, usize)>,
) -> Option<&EdgeGroup> {
    let (vertex, index) = link?;
    groups.get(vertex)?.as_ref()?.get(index)
}

fn invariant(message: &'static str) -> PotError {
    PotError::new(ErrorCode::InternalError, message)
}

/// Read-only inputs of the vertex-position solve (Blender's `MOD_solidify_nonmanifold` locals).
struct Geo<'a> {
    /// Merged vertex positions (`orig_mvert_co`).
    co: &'a [V3],
    vm: &'a [usize],
    orig_edges: &'a [[usize; 2]],
    corner_verts: &'a [usize],
    sides: &'a [Side],
    nedges: &'a [NewEdge],
    face_nors: &'a [V3],
    null_faces: &'a [bool],
    /// Per-face scale for `use_flat_faces`; present exactly when that option is active.
    face_weight: Option<&'a [f32]>,
    orig_edge_lengths: &'a [f32],
    /// Raw thickness-group weights (`dvert`), present when the group exists.
    vgroup: Option<&'a [f32]>,
    invert_vgroup: bool,
    offset_fac: f32,
    offset_fac_vg: f32,
    offset_fac_vg_inv: f32,
    ofs_front: f32,
    ofs_back: f32,
    /// `offset`: `abs(thickness) * thickness_clamp`.
    offset: f32,
    do_clamp: bool,
    do_angle_clamp: bool,
    thickness_mode: ThicknessMode,
    boundary_mode: BoundaryMode,
}

/// Offset position of one edge group (the group part of `MOD_solidify_nonmanifold_modifyMesh`).
fn group_coord(geo: &Geo, i: usize, group: &EdgeGroup) -> V3 {
    let base = geo.co[i];
    if group.is_singularity {
        return base;
    }
    let edges = &group.edges;
    let n = edges.len();
    let cycle = (group.is_orig_closed && group.split == 0) || group.is_even_split;
    let mut nor: V3 = [0.0; 3];
    let mut move_nor: V3 = [0.0; 3];
    let mut disable_boundary_fix =
        geo.boundary_mode == BoundaryMode::None || group.is_orig_closed || group.split != 0;
    let mut approximate_free_direction = false;

    if geo.thickness_mode == ThicknessMode::Constraints {
        let mut first_edge = EMPTY;
        let mut planes: Vec<[f32; 4]> = Vec::with_capacity(n + 1);
        let mut fallback_nor: V3 = [0.0; 3];
        let mut fallback_ofs = 0.0_f32;
        for (k, &edge) in edges.iter().enumerate() {
            let sampled = k % 2 == 0 || (!cycle && k == n - 1);
            if !sampled {
                continue;
            }
            for side_slot in 0..2 {
                let face = geo.nedges[edge].faces[side_slot];
                if face == EMPTY
                    || (first_edge != EMPTY
                        && (geo.nedges[first_edge].faces[0] == face
                            || geo.nedges[first_edge].faces[1] == face))
                {
                    continue;
                }
                let side = &geo.sides[face];
                let mut ofs = if side.reversed {
                    geo.ofs_back
                } else {
                    geo.ofs_front
                };
                if let Some(weights) = geo.face_weight {
                    ofs *= weights[side.face];
                }
                let sign = if side.reversed { -1.0 } else { 1.0 };
                let normal = mul3(geo.face_nors[side.face], sign);
                if geo.null_faces[side.face] {
                    fallback_nor = normal;
                    fallback_ofs = ofs;
                } else {
                    planes.push([normal[0], normal[1], normal[2], ofs]);
                }
            }
            if (cycle && k == 0) || (!cycle && k + 3 >= n) {
                first_edge = edge;
            }
        }
        let n3 = |p: &[f32; 4]| -> V3 { [p[0], p[1], p[2]] };
        if planes.len() > 2 {
            // Find the two most different normals and move them to the front.
            let mut min_p = 2.0_f32;
            let mut min_n0 = 0;
            let mut min_n1 = 0;
            for k in 0..planes.len() {
                for m in k + 1..planes.len() {
                    let p = dot3(n3(&planes[k]), n3(&planes[m]));
                    if p < min_p {
                        min_p = p;
                        min_n0 = k;
                        min_n1 = m;
                    }
                }
            }
            if min_n1 != 0 {
                planes.swap(min_n0, 0);
                planes.swap(min_n1, 1);
            } else {
                planes.swap(min_n0, 1);
            }
            // Find the third most important/different normal.
            min_p = 1.0;
            min_n1 = 2;
            for k in 2..planes.len() {
                let max_p = max_ff(
                    dot3(n3(&planes[0]), n3(&planes[k])),
                    dot3(n3(&planes[1]), n3(&planes[k])),
                );
                if max_p <= min_p {
                    min_p = max_p;
                    min_n1 = k;
                }
            }
            planes.swap(min_n1, 2);
        }
        // Merge duplicate normals, averaging their normals and offsets.
        while planes.len() > 2 {
            let mut best_n0 = 0;
            let mut best_n1 = 0;
            let mut best_p = -1.0_f32;
            let mut best_ofs_diff = 0.0_f32;
            for k in 0..planes.len() {
                for m in k + 1..planes.len() {
                    let p = dot3(n3(&planes[m]), n3(&planes[k]));
                    let ofs_diff = (planes[m][3] - planes[k][3]).abs();
                    if p > best_p + EPS || (p >= best_p && ofs_diff < best_ofs_diff) {
                        best_p = p;
                        best_ofs_diff = ofs_diff;
                        best_n0 = k;
                        best_n1 = m;
                    }
                }
            }
            // Keep the threshold: it is what makes the methods below numerically stable.
            if best_p < 0.98 {
                break;
            }
            let mut merged = add3(n3(&planes[best_n0]), n3(&planes[best_n1]));
            normalize3(&mut merged);
            let ofs = blender_midpoint(planes[best_n0][3], planes[best_n1][3]);
            planes[best_n0] = [merged[0], merged[1], merged[2], ofs];
            planes.remove(best_n1);
        }
        let size = planes.len();
        let boundary_fix_threshold = 0.7_f32;
        if size > 3 {
            // Least squares position: solve (sum n nT + 5e-5 I) x = sum n * ofs.
            let mut mat = [[0.0_f32; 3]; 3];
            for k in 0..3 {
                for plane in &planes {
                    for r in 0..3 {
                        mat[r][k] += plane[r] * plane[k];
                    }
                }
                mat[k][k] += 5.0e-5;
            }
            let inv = invert3(mat);
            let mut rhs: V3 = [0.0; 3];
            for plane in &planes {
                rhs = add3(rhs, mul3(n3(plane), plane[3]));
            }
            nor = [
                inv[0][0] * rhs[0] + inv[0][1] * rhs[1] + inv[0][2] * rhs[2],
                inv[1][0] * rhs[0] + inv[1][1] * rhs[1] + inv[1][2] * rhs[2],
                inv[2][0] * rhs[0] + inv[2][1] * rhs[1] + inv[2][2] * rhs[2],
            ];
            if !disable_boundary_fix {
                let mut greatest_angle_cos = 1.0_f32;
                for k in 0..2 {
                    for m in 2..size {
                        let p = dot3(n3(&planes[m]), n3(&planes[k]));
                        greatest_angle_cos = if p < greatest_angle_cos {
                            p
                        } else {
                            greatest_angle_cos
                        };
                    }
                }
                if greatest_angle_cos > boundary_fix_threshold {
                    approximate_free_direction = true;
                } else {
                    disable_boundary_fix = true;
                }
            }
        } else if size > 1 {
            // Up to three constraint planes have a closed-form solution.
            let stop_explosion = 0.999 - geo.offset_fac.abs() * 0.05;
            let q = dot3(n3(&planes[0]), n3(&planes[1]));
            let mut d = 1.0 - q * q;
            move_nor = cross3(n3(&planes[0]), n3(&planes[1]));
            normalize3(&mut move_nor);
            let (ofs0, ofs1) = (planes[0][3], planes[1][3]);
            if d > EPS * 10.0 && q < stop_explosion {
                d = 1.0 / d;
                let scale0 = (ofs0 - ofs1 * q) * d;
                let scale1 = (ofs1 - ofs0 * q) * d;
                planes[0] = [
                    planes[0][0] * scale0,
                    planes[0][1] * scale0,
                    planes[0][2] * scale0,
                    ofs0,
                ];
                planes[1] = [
                    planes[1][0] * scale1,
                    planes[1][1] * scale1,
                    planes[1][2] * scale1,
                    ofs1,
                ];
            } else {
                d = 1.0 / (q.abs() + 1.0);
                let scale0 = ofs0 * d;
                let scale1 = ofs1 * d;
                planes[0] = [
                    planes[0][0] * scale0,
                    planes[0][1] * scale0,
                    planes[0][2] * scale0,
                    ofs0,
                ];
                planes[1] = [
                    planes[1][0] * scale1,
                    planes[1][1] * scale1,
                    planes[1][2] * scale1,
                    ofs1,
                ];
            }
            nor = add3(n3(&planes[0]), n3(&planes[1]));
            if size == 3 {
                d = dot3(n3(&planes[2]), move_nor);
                // Ignore a third plane that is almost orthogonal to the still free direction.
                if d.abs() > 0.02 {
                    let tmp = add3(nor, mul3(n3(&planes[2]), -planes[2][3]));
                    let correction = mul3(move_nor, dot3(n3(&planes[2]), tmp) / d);
                    nor = sub3(nor, correction);
                    if d.abs() > 1.0 - boundary_fix_threshold {
                        disable_boundary_fix = true;
                    }
                }
            }
            approximate_free_direction = false;
        } else if size == 1 {
            // Face corner.
            nor = mul3(n3(&planes[0]), planes[0][3]);
            if n > 2 {
                disable_boundary_fix = true;
                approximate_free_direction = true;
            }
        } else {
            // Null faces only.
            nor = mul3(fallback_nor, fallback_ofs);
            disable_boundary_fix = true;
        }
    } else {
        let even = geo.thickness_mode == ThicknessMode::Even;
        let mut total_angle = 0.0_f32;
        let mut total_angle_back = 0.0_f32;
        let mut first_edge = EMPTY;
        let mut nor_back: V3 = [0.0; 3];
        let mut has_back = false;
        let mut has_front = false;
        for (k, &edge) in edges.iter().enumerate() {
            let sampled = k % 2 == 0 || (!cycle && k == n - 1);
            if !sampled {
                continue;
            }
            for side_slot in 0..2 {
                let face = geo.nedges[edge].faces[side_slot];
                if face == EMPTY
                    || (first_edge != EMPTY
                        && (geo.nedges[first_edge].faces[0] == face
                            || geo.nedges[first_edge].faces[1] == face))
                {
                    continue;
                }
                let side = &geo.sides[face];
                let mut angle = 1.0_f32;
                let mut ofs = if side.reversed {
                    -geo.ofs_back
                } else {
                    geo.ofs_front
                };
                if let Some(weights) = geo.face_weight {
                    ofs *= weights[side.face];
                }
                if even {
                    // Angle at vertex `i` inside this face, found by walking its corners.
                    let mut corner_next = side.start;
                    let mut corner = side.start + side.size - 1;
                    let mut corner_prev = corner - 1;
                    let mut m = 0;
                    while m < side.size && geo.vm[geo.corner_verts[corner]] != i {
                        corner_prev = corner;
                        corner = corner_next;
                        corner_next += 1;
                        m += 1;
                    }
                    angle = angle_v3v3v3(
                        geo.co[geo.vm[geo.corner_verts[corner_prev]]],
                        geo.co[i],
                        geo.co[geo.vm[geo.corner_verts[corner_next]]],
                    );
                    if side.reversed {
                        total_angle_back += angle * ofs * ofs;
                    } else {
                        total_angle += angle * ofs * ofs;
                    }
                } else if side.reversed {
                    total_angle_back += 1.0;
                } else {
                    total_angle += 1.0;
                }
                let face_nor = mul3(geo.face_nors[side.face], angle * ofs);
                if side.reversed {
                    nor_back = add3(nor_back, face_nor);
                    has_back = true;
                } else {
                    nor = add3(nor, face_nor);
                    has_front = true;
                }
            }
            if (cycle && k == 0) || (!cycle && k + 3 >= n) {
                first_edge = edge;
            }
        }

        if even {
            if has_front {
                let length_sq = dot3(nor, nor);
                if length_sq > EPS {
                    nor = mul3(nor, total_angle / length_sq);
                }
            }
            if has_back {
                let length_sq = dot3(nor_back, nor_back);
                if length_sq > EPS {
                    nor_back = mul3(nor_back, total_angle_back / length_sq);
                }
                if !has_front {
                    nor = nor_back;
                }
            }
            if has_front && has_back {
                let nor_length = len3(nor);
                let nor_back_length = len3(nor_back);
                let mut q = dot3(nor, nor_back);
                if q.abs() > EPS {
                    q /= nor_length * nor_back_length;
                }
                let d = 1.0 - q * q;
                if d > EPS {
                    let inv_d = 1.0 / d;
                    if nor_length > EPS {
                        nor = mul3(nor, (1.0 - nor_back_length * q / nor_length) * inv_d);
                    }
                    if nor_back_length > EPS {
                        nor_back = mul3(nor_back, (1.0 - nor_length * q / nor_back_length) * inv_d);
                    }
                    nor = add3(nor, nor_back);
                } else {
                    nor = add3(mul3(nor, 0.5), mul3(nor_back, 0.5));
                }
            }
        } else {
            if has_front && total_angle > EPS {
                nor = mul3(nor, 1.0 / total_angle);
            }
            if has_back && total_angle_back > EPS {
                nor_back = mul3(nor_back, 1.0 / total_angle_back);
                nor = add3(nor, nor_back);
                if has_front && total_angle > EPS {
                    nor = mul3(nor, 0.5);
                }
            }
        }
        if !disable_boundary_fix && n > 2 {
            approximate_free_direction = true;
        } else {
            disable_boundary_fix = true;
        }
    }

    if approximate_free_direction {
        // Free direction from the edges between the first and last edge of the group.
        let mut k = 1;
        while k + 1 < n {
            let edge = geo.orig_edges[geo.nedges[edges[k]].old];
            let other = if geo.vm[edge[0]] == i {
                edge[1]
            } else {
                edge[0]
            };
            move_nor = add3(move_nor, sub3(geo.co[other], geo.co[i]));
            k += 1;
        }
        if k == 1 {
            disable_boundary_fix = true;
        } else {
            disable_boundary_fix = normalize3(&mut move_nor) == 0.0;
        }
    }

    if !disable_boundary_fix {
        // Keep `nor` orthogonal to the boundary constraint, as far as the thickness allows.
        let first = geo.orig_edges[geo.nedges[edges[0]].old];
        let last = geo.orig_edges[geo.nedges[edges[n - 1]].old];
        let e0 = sub3(
            geo.co[if geo.vm[first[0]] == i {
                first[1]
            } else {
                first[0]
            }],
            geo.co[i],
        );
        let e1 = sub3(
            geo.co[if geo.vm[last[0]] == i {
                last[1]
            } else {
                last[0]
            }],
            geo.co[i],
        );
        let mut constr_nor: V3;
        if geo.boundary_mode == BoundaryMode::Flat {
            constr_nor = cross3(e0, e1);
            normalize3(&mut constr_nor);
        } else {
            let face0 = &geo.sides[geo.nedges[edges[0]].faces[0]];
            let face1 = &geo.sides[geo.nedges[edges[n - 1]].faces[0]];
            let f0 = if face0.reversed {
                negate3(geo.face_nors[face0.face])
            } else {
                geo.face_nors[face0.face]
            };
            let f1 = if face1.reversed {
                negate3(geo.face_nors[face1.face])
            } else {
                geo.face_nors[face1.face]
            };
            let mut n0 = cross3(e0, f0);
            let mut n1 = cross3(f1, e1);
            normalize3(&mut n0);
            normalize3(&mut n1);
            constr_nor = add3(n0, n1);
            normalize3(&mut constr_nor);
        }
        let d = dot3(constr_nor, move_nor);
        // Only allow the thickness to increase about 10 times.
        if d.abs() > 0.1 {
            move_nor = mul3(move_nor, dot3(constr_nor, nor) / d);
            nor = sub3(nor, move_nor);
        }
    }

    let mut scalar = 1.0_f32;
    if geo.face_weight.is_none()
        && let Some(weights) = geo.vgroup
    {
        let weight = weights[i];
        scalar = if geo.invert_vgroup {
            1.0 - weight
        } else {
            weight
        };
        scalar = geo.offset_fac_vg + scalar * geo.offset_fac_vg_inv;
    }
    if geo.do_clamp {
        if geo.do_angle_clamp {
            if n > 2 {
                let mut min_length = 0.0_f32;
                let mut angle = (0.5 * PI_F64) as f32;
                for (k, &edge) in edges.iter().enumerate() {
                    let length = geo.orig_edge_lengths[geo.nedges[edge].old];
                    let e_ang = geo.nedges[edge].angle;
                    angle = max_std(e_ang, angle);
                    if length < min_length || k == 0 {
                        min_length = length;
                    }
                }
                let cos_ang = (angle * 0.5).cos();
                if cos_ang > 0.0 {
                    let max_off = min_length * 0.5 / cos_ang;
                    if max_off < geo.offset * 0.5 {
                        scalar *= max_off / geo.offset * 2.0;
                    }
                }
            }
        } else {
            let mut min_length = 0.0_f32;
            for (k, &edge) in edges.iter().enumerate() {
                let length = geo.orig_edge_lengths[geo.nedges[edge].old];
                if length < min_length || k == 0 {
                    min_length = length;
                }
            }
            if min_length < geo.offset {
                scalar *= min_length / geo.offset;
            }
        }
    }
    add3(mul3(nor, scalar), base)
}

/// Blender `MOD_solidify_nonmanifold_modifyMesh` topology: original geometry after the vertex merge,
/// the sides and new edges around each original edge, and the edge groups around each vertex.
struct Topo {
    sides: Vec<Side>,
    nedges: Vec<NewEdge>,
    /// `orig_edge_data_arr`: index into `edge_lists` of the new edges of an original edge.
    orig_edge_data: Vec<Option<usize>>,
    edge_lists: Vec<Vec<usize>>,
    /// `edge_adj_faces_len`: adjacent-face count of each original edge after merges; zero when unused.
    edge_len: Vec<usize>,
    /// `vm`: merged vertex representative of each original vertex.
    vm: Vec<usize>,
    /// `orig_mvert_co`: merged positions, indexed by original vertex.
    co: Vec<V3>,
    orig_edge_lengths: Vec<f32>,
    /// `orig_vert_groups_arr`: edge groups per original vertex; `None` when fewer than two edges.
    vert_groups: Vec<Option<Vec<EdgeGroup>>>,
    has_singularities: bool,
    new_verts: usize,
}

/// Original mesh topology in index form.
struct Corners {
    face_start: Vec<usize>,
    face_size: Vec<usize>,
    /// `orig_corner_verts` in face order.
    corner_verts: Vec<usize>,
    /// `orig_corner_edges` in face order.
    corner_edges: Vec<usize>,
}

/// Blender `MOD_solidify_nonmanifold_modifyMesh` up to the vertex positions: merges, face-set
/// adjacency, new edges and edge groups.
fn build_topology(
    corners: &Corners,
    edge_list: &[[usize; 2]],
    co_in: &[V3],
    face_nors: &[V3],
    do_shell: bool,
    do_rim: bool,
    merge_threshold: f32,
) -> Result<Topo> {
    let verts_num = co_in.len();
    let edges_num = edge_list.len();
    let faces_num = corners.face_size.len();
    let mut sides: Vec<Side> = Vec::with_capacity(faces_num * 2);
    for face in 0..faces_num {
        for reversed in [false, true] {
            sides.push(Side {
                face,
                reversed,
                start: corners.face_start[face],
                size: corners.face_size[face],
                links: vec![EMPTY; corners.face_size[face]],
            });
        }
    }

    // Faces around each edge, with each face's orientation relative to the edge direction.
    let mut edge_len = vec![0_usize; edges_num];
    for &edge in &corners.corner_edges {
        edge_len[edge] += 1;
    }
    let mut adj: Vec<AdjFaces> = Vec::new();
    let mut edge_adj: Vec<Option<usize>> = vec![None; edges_num];
    for face in 0..faces_num {
        let start = corners.face_start[face];
        for corner in start..start + corners.face_size[face] {
            let vert = corners.corner_verts[corner];
            let edge = corners.corner_edges[corner];
            let reversed = edge_list[edge][1] != vert;
            match edge_adj[edge] {
                None => {
                    let mut faces = vec![EMPTY; edge_len[edge]];
                    let mut flags = vec![false; edge_len[edge]];
                    faces[0] = face;
                    flags[0] = reversed;
                    adj.push(AdjFaces {
                        faces,
                        reversed: flags,
                        used: 1,
                    });
                    edge_adj[edge] = Some(adj.len() - 1);
                }
                Some(id) => {
                    let object = &mut adj[id];
                    for k in 1..object.faces.len() {
                        if object.faces[k] == EMPTY {
                            object.faces[k] = face;
                            object.reversed[k] = reversed;
                            break;
                        }
                    }
                }
            }
        }
    }

    // Merge edges shorter than the threshold, unless that would change the face count.
    let mut vm: Vec<usize> = (0..verts_num).collect();
    let mut co: Vec<V3> = co_in.to_vec();
    let mut vert_adj_len = vec![0_usize; verts_num];
    let mut combined = vec![0_usize; verts_num];
    let mut orig_edge_lengths = vec![0.0_f32; edges_num];
    let merge_sq = merge_threshold * merge_threshold;
    for i in 0..edges_num {
        if edge_len[i] == 0 {
            continue;
        }
        let [e0, e1] = edge_list[i];
        let mut v1 = vm[e0];
        let mut v2 = vm[e1];
        if v1 == v2 {
            continue;
        }
        if v2 < v1 {
            std::mem::swap(&mut v1, &mut v2);
        }
        let mut edgedir = sub3(co[v2], co[v1]);
        orig_edge_lengths[i] = dot3(edgedir, edgedir);
        if orig_edge_lengths[i] <= merge_sq {
            let mut can_merge = true;
            for k in 0..edges_num {
                if !can_merge {
                    break;
                }
                if k == i || edge_len[k] == 0 {
                    continue;
                }
                let [k0, k1] = edge_list[k];
                if is_elem(vm[k0], v1, v2) == is_elem(vm[k1], v1, v2) {
                    continue;
                }
                let Some(id) = edge_adj[k] else {
                    continue;
                };
                for j in 0..adj[id].faces.len() {
                    if !can_merge {
                        break;
                    }
                    let face = adj[id].faces[j];
                    let start = corners.face_start[face];
                    let size = corners.face_size[face];
                    let mut changes = 0_usize;
                    let mut has_multiple_unique_others = false;
                    let mut unique_other_vert = EMPTY;
                    let mut cur = size - 1;
                    let mut next = 0;
                    while next < size && changes <= 2 {
                        let cur_v = vm[corners.corner_verts[start + cur]];
                        let next_v = vm[corners.corner_verts[start + next]];
                        changes += usize::from(is_elem(cur_v, v1, v2) != is_elem(next_v, v1, v2));
                        if !is_elem(cur_v, v1, v2) {
                            if unique_other_vert == EMPTY {
                                unique_other_vert = cur_v;
                            } else if unique_other_vert != cur_v {
                                has_multiple_unique_others = true;
                            }
                        }
                        cur = next;
                        next += 1;
                    }
                    can_merge =
                        (has_multiple_unique_others || changes != 2) && changes <= 2 && can_merge;
                }
            }
            if !can_merge {
                orig_edge_lengths[i] = 0.0;
                vert_adj_len[v1] += 1;
                vert_adj_len[v2] += 1;
                continue;
            }
            let scale = (combined[v2] + 1) as f32 / (combined[v1] + combined[v2] + 2) as f32;
            edgedir = mul3(edgedir, scale);
            co[v1] = add3(co[v1], edgedir);
            for mapped in vm.iter_mut().skip(v2) {
                if *mapped == v2 {
                    *mapped = v1;
                }
            }
            vert_adj_len[v1] += vert_adj_len[v2];
            vert_adj_len[v2] = 0;
            combined[v1] += combined[v2] + 1;
            edge_len[i] = 0;
            edge_adj[i] = None;
        } else {
            orig_edge_lengths[i] = orig_edge_lengths[i].sqrt();
            vert_adj_len[v1] += 1;
            vert_adj_len[v2] += 1;
        }
    }
    // Edges whose ends merged into one vertex are collapsed away.
    for i in 0..edges_num {
        let [e0, e1] = edge_list[i];
        if vm[e0] == vm[e1] && edge_adj[i].is_some() {
            edge_len[i] = 0;
            edge_adj[i] = None;
        }
    }

    // Vertex-to-edge lists. An edge that duplicates an existing one takes over its faces, and faces
    // that used both become zero-area and are dropped.
    let mut vert_adj: Vec<Option<VertEdges>> = (0..verts_num).map(|_| None).collect();
    for i in 0..edges_num {
        if edge_len[i] == 0 {
            continue;
        }
        let [e0, e1] = edge_list[i];
        let vs = [vm[e0], vm[e1]];
        let mut invalid: Option<usize> = None;
        let mut invalid_reversed = false;
        for j in 0..2 {
            let vert = vs[j];
            let len = vert_adj_len[vert];
            if len == 0 {
                continue;
            }
            match vert_adj[vert].as_mut() {
                None => {
                    let mut edges = vec![EMPTY; len];
                    edges[0] = i;
                    vert_adj[vert] = Some(VertEdges {
                        edges,
                        edges_len: 1,
                    });
                }
                Some(list) => {
                    let mut k = 0;
                    while k < len && k <= list.edges_len {
                        let edge = list.edges[k];
                        if edge == EMPTY || k == list.edges_len {
                            list.edges[k] = i;
                            list.edges_len += 1;
                            break;
                        }
                        if vm[edge_list[edge][0]] == vs[1 - j] {
                            invalid = Some(edge);
                            invalid_reversed = j == 0;
                            break;
                        }
                        if vm[edge_list[edge][1]] == vs[1 - j] {
                            invalid = Some(edge);
                            invalid_reversed = j == 1;
                            break;
                        }
                        k += 1;
                    }
                }
            }
            if invalid.is_some() {
                if j == 1
                    && let Some(first) = vert_adj[vs[0]].as_mut()
                {
                    first.edges_len = first.edges_len.saturating_sub(1);
                }
                break;
            }
        }
        let Some(existing) = invalid else {
            continue;
        };
        // Edge `i` duplicates `existing`: merge its faces into `existing`'s object.
        let (Some(keep_id), Some(drop_id)) = (edge_adj[existing], edge_adj[i]) else {
            return Err(invariant("solidify duplicate edge has no face set"));
        };
        let mut keep = std::mem::replace(
            &mut adj[keep_id],
            AdjFaces {
                faces: Vec::new(),
                reversed: Vec::new(),
                used: 0,
            },
        );
        let dropped = &mut adj[drop_id];
        for k in 0..keep.faces.len() {
            for l in 0..dropped.faces.len() {
                if keep.faces[k] == dropped.faces[l] && keep.faces[k] != EMPTY {
                    keep.faces[k] = EMPTY;
                    dropped.faces[l] = EMPTY;
                }
            }
        }
        let mut faces = Vec::new();
        let mut flags = Vec::new();
        for k in 0..keep.faces.len() {
            if keep.faces[k] != EMPTY {
                faces.push(keep.faces[k]);
                flags.push(keep.reversed[k]);
            }
        }
        for k in 0..dropped.faces.len() {
            if dropped.faces[k] != EMPTY {
                faces.push(dropped.faces[k]);
                flags.push(invalid_reversed != dropped.reversed[k]);
            }
        }
        edge_len[i] = 0;
        edge_len[existing] = faces.len();
        keep.faces = faces;
        keep.reversed = flags;
        keep.used += dropped.used;
        dropped.faces.clear();
        dropped.reversed.clear();
        adj[keep_id] = keep;
        edge_adj[i] = Some(keep_id);
    }
    // Faces that repeat an earlier face's vertex loop on one edge are dropped.
    for i in 0..edges_num {
        if edge_len[i] == 0 {
            continue;
        }
        let Some(adj_id) = edge_adj[i] else {
            continue;
        };
        let mut adj_len = adj[adj_id].faces.len();
        if adj_len > 1 {
            let mut j = 0;
            while j < adj_len {
                let face = adj[adj_id].faces[j];
                let j_loopstart = corners.face_start[face];
                let totloop = corners.face_size[face];
                let j_first_v = vm[corners.corner_verts[j_loopstart]];
                let mut k = j + 1;
                while k < adj_len {
                    let other = adj[adj_id].faces[k];
                    if corners.face_size[other] != totloop {
                        k += 1;
                        continue;
                    }
                    let k_loopstart = corners.face_start[other];
                    let Some(l) = (0..totloop)
                        .find(|&l| vm[corners.corner_verts[k_loopstart + l]] == j_first_v)
                    else {
                        k += 1;
                        continue;
                    };
                    let reversed = adj[adj_id].reversed[j] != adj[adj_id].reversed[k];
                    let mut has_diff = false;
                    for m in 0..totloop {
                        if has_diff {
                            break;
                        }
                        let vert = corners.corner_verts[j_loopstart + m];
                        let other_corner = if reversed {
                            (l + totloop - m) % totloop
                        } else {
                            (l + m) % totloop
                        };
                        has_diff = vm[vert] != vm[corners.corner_verts[k_loopstart + other_corner]];
                    }
                    if !has_diff {
                        for m in 0..totloop {
                            let e = corners.corner_edges[j_loopstart + m];
                            let Some(e_obj) = edge_adj[e] else {
                                continue;
                            };
                            let face_index = if e_obj == adj_id {
                                adj_len -= 1;
                                j
                            } else {
                                let Some(position) =
                                    adj[e_obj].faces.iter().position(|&f| f == face)
                                else {
                                    continue;
                                };
                                position
                            };
                            adj[e_obj].faces.remove(face_index);
                            adj[e_obj].reversed.remove(face_index);
                            if edge_len[e] > 0 {
                                edge_len[e] -= 1;
                                if edge_len[e] == 0 {
                                    adj[e_obj].used = adj[e_obj].used.saturating_sub(1);
                                    edge_adj[e] = None;
                                }
                            } else if adj[e_obj].used > 1 {
                                for n in 0..edges_num {
                                    if edge_adj[n] == Some(e_obj) && edge_len[n] > 0 {
                                        edge_len[n] -= 1;
                                        if edge_len[n] == 0 {
                                            adj[e_obj].used = adj[e_obj].used.saturating_sub(1);
                                            edge_adj[n] = None;
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                        break;
                    }
                    k += 1;
                }
                j += 1;
            }
        }
    }

    // One new edge per side pair around each original edge, sorted by angle around the edge.
    let mut nedges: Vec<NewEdge> = Vec::new();
    let mut orig_edge_data: Vec<Option<usize>> = vec![None; edges_num];
    let mut edge_lists: Vec<Vec<usize>> = Vec::new();
    for i in 0..edges_num {
        if edge_len[i] == 0 {
            continue;
        }
        let Some(adj_id) = edge_adj[i] else {
            return Err(invariant("solidify edge with faces has no face set"));
        };
        let [e0, e1] = edge_list[i];
        let v1 = vm[e0];
        let v2 = vm[e1];
        let edgedir = if orig_edge_lengths[i] > EPS {
            mul3(sub3(co[v2], co[v1]), 1.0 / orig_edge_lengths[i])
        } else {
            // A degenerate edge takes its direction from the neighbouring edges at both ends.
            let pos = co[v2];
            let mut v1_dir: V3 = [0.0; 3];
            if let Some(link) = &vert_adj[v1] {
                for &e in &link.edges[..link.edges_len] {
                    if edge_len[e] > 0 && e != i {
                        let other = vm[if vm[edge_list[e][0]] == v1 {
                            edge_list[e][1]
                        } else {
                            edge_list[e][0]
                        }];
                        v1_dir = add3(v1_dir, sub3(co[other], pos));
                    }
                }
            }
            let mut v2_dir: V3 = [0.0; 3];
            if let Some(link) = &vert_adj[v2] {
                for &e in &link.edges[..link.edges_len] {
                    if edge_len[e] > 0 && e != i {
                        let other = vm[if vm[edge_list[e][0]] == v2 {
                            edge_list[e][1]
                        } else {
                            edge_list[e][0]
                        }];
                        v2_dir = add3(v2_dir, sub3(co[other], pos));
                    }
                }
            }
            let mut dir = sub3(v2_dir, v1_dir);
            if normalize3(&mut dir) == 0.0 {
                dir = [0.0, 0.0, 1.0];
            }
            dir
        };
        let obj_faces = adj[adj_id].faces.clone();
        let obj_reversed = adj[adj_id].reversed.clone();
        let adj_len = obj_faces.len();
        let mut sorted: Vec<(f32, usize)> = Vec::with_capacity(adj_len.max(2));
        let new_edges_len = if adj_len > 1 {
            let mut ref_nor: V3 = [0.0; 3];
            for j in 0..adj_len {
                let reverse = obj_reversed[j];
                let face_i = obj_faces[j];
                let mut nor = if reverse {
                    negate3(face_nors[face_i])
                } else {
                    face_nors[face_i]
                };
                let mut d = 1.0_f32;
                if corners.face_size[face_i] > 3 {
                    d = project3(&mut nor, edgedir);
                    d = if d == 0.0 { 1.0 } else { normalize3(&mut nor) };
                }
                let angle = if d == 0.0 {
                    0.0
                } else if j == 0 {
                    ref_nor = nor;
                    0.0
                } else {
                    -angle_signed(nor, ref_nor, edgedir)
                };
                sorted.push((angle, face_i * 2 + usize::from(reverse)));
            }
            // Stable, so faces with equal keys keep their order.
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            adj_len
        } else {
            sorted.push((0.0, obj_faces[0] * 2 + usize::from(obj_reversed[0])));
            2
        };
        let list_id = edge_lists.len();
        edge_lists.push(Vec::new());
        let marker = if do_shell || (adj_len == 1 && do_rim) {
            0
        } else {
            EMPTY
        };
        let mut new_edges = Vec::with_capacity(new_edges_len);
        for j in 0..new_edges_len {
            let (faces, angle) = if adj_len > 1 {
                let next_j = if j + 1 == adj_len { 0 } else { j + 1 };
                let mut angle = sorted[next_j].0 - sorted[j].0;
                if angle < 0.0 {
                    angle = (f64::from(angle) + 2.0 * PI_F64) as f32;
                }
                ([sorted[j].1, sorted[next_j].1 ^ 1], angle)
            } else {
                ([sorted[0].1 ^ j, EMPTY], 0.0)
            };
            let id = nedges.len();
            nedges.push(NewEdge {
                old: i,
                faces,
                links: [None, None],
                angle,
                output_edge: marker,
            });
            new_edges.push(id);
            for &face in &faces {
                if face == EMPTY {
                    continue;
                }
                let start = sides[face].start;
                for l in 0..sides[face].size {
                    let edge = corners.corner_edges[start + l];
                    if edge_adj[edge].is_some() && edge_adj[edge] == edge_adj[i] {
                        if edge != i && orig_edge_data[edge].is_none() {
                            orig_edge_data[edge] = Some(list_id);
                        }
                        sides[face].links[l] = id;
                        break;
                    }
                }
            }
        }
        edge_lists[list_id] = new_edges;
        orig_edge_data[i] = Some(list_id);
    }

    // Edge groups: the new edges around each vertex, joined into consecutive runs.
    let mut vert_groups: Vec<Option<Vec<EdgeGroup>>> = vec![None; verts_num];
    let mut has_singularities = false;
    let mut new_verts = 0_usize;
    for i in 0..verts_num {
        let Some(list) = vert_adj[i].as_ref() else {
            continue;
        };
        if list.edges_len < 2 {
            continue;
        }
        let mut unassigned: Vec<Option<usize>> = Vec::new();
        for &e in &list.edges[..list.edges_len] {
            if let Some(data) = orig_edge_data[e] {
                unassigned.extend(edge_lists[data].iter().map(|&id| Some(id)));
            }
        }
        let unassigned_len = unassigned.len();
        let mut groups = vec![EdgeGroup::default(); unassigned_len * 2 + 4];
        let mut eg: isize = -1;
        let mut topo_groups = 0_usize;
        let mut contains_long = false;
        let mut track = [EMPTY; 2];
        let mut last_open = EMPTY;
        let mut assigned = 0_usize;
        while assigned < unassigned_len {
            let mut found: Option<usize> = None;
            let mut found_index = 0_usize;
            let mut insert_at_start = false;
            if eg >= 0 && groups[eg as usize].edges.is_empty() {
                // Start a new run: an edge that continues the open run if there is one.
                let g = eg as usize;
                let mut j = 0;
                let mut edge: Option<usize> = None;
                while edge.is_none() && j < unassigned_len {
                    edge = unassigned[j];
                    j += 1;
                    if let Some(e) = edge
                        && last_open != EMPTY
                        && (nedges[e].faces[0] != last_open || nedges[e].faces[1] != EMPTY)
                    {
                        edge = None;
                    }
                }
                if edge.is_none() && last_open != EMPTY {
                    topo_groups += 1;
                    last_open = EMPTY;
                    groups[g].topo_group += 1;
                    j = 0;
                    while edge.is_none() && j < unassigned_len {
                        edge = unassigned[j];
                        j += 1;
                    }
                } else if last_open == EMPTY && eg > 0 {
                    topo_groups += 1;
                    groups[g].topo_group += 1;
                }
                let Some(e) = edge else {
                    return Err(invariant("solidify edge group found no starting edge"));
                };
                found_index = j - 1;
                found = Some(e);
                if last_open == EMPTY && vm[edge_list[nedges[e].old][0]] == i {
                    track = nedges[e].faces;
                    if nedges[e].faces[1] == EMPTY {
                        last_open = nedges[e].faces[0] ^ 1;
                    }
                } else {
                    track = [nedges[e].faces[1], nedges[e].faces[0]];
                }
            } else if eg >= 0 {
                let g = eg as usize;
                for (index, edge) in unassigned.iter().enumerate() {
                    found_index = index;
                    let Some(e) = *edge else {
                        continue;
                    };
                    let faces = nedges[e].faces;
                    if faces[0] == track[1] {
                        insert_at_start = false;
                        track[1] = faces[1];
                        found = Some(e);
                        if faces[1] == EMPTY {
                            groups[g].is_orig_closed = false;
                            last_open = faces[0] ^ 1;
                        }
                        break;
                    }
                    if faces[0] == track[0] {
                        insert_at_start = true;
                        track[0] = faces[1];
                        found = Some(e);
                        if faces[1] == EMPTY {
                            groups[g].is_orig_closed = false;
                        }
                        break;
                    }
                    if faces[1] != EMPTY {
                        if faces[1] == track[1] {
                            insert_at_start = false;
                            track[1] = faces[0];
                            found = Some(e);
                            break;
                        }
                        if faces[1] == track[0] {
                            insert_at_start = true;
                            track[0] = faces[0];
                            found = Some(e);
                            break;
                        }
                    }
                }
            }
            if let Some(e) = found {
                unassigned[found_index] = None;
                assigned += 1;
                let g = eg as usize;
                if insert_at_start {
                    groups[g].edges.insert(0, e);
                } else {
                    groups[g].edges.push(e);
                }
                if let Some(&last) = groups[g].edges.last()
                    && nedges[last].faces[1] != EMPTY
                {
                    last_open = EMPTY;
                }
                if groups[g].edges.len() > 3 {
                    contains_long = true;
                }
            } else {
                eg += 1;
                let g = eg as usize;
                groups[g] = EdgeGroup {
                    valid: true,
                    topo_group: topo_groups,
                    is_orig_closed: true,
                    ..EdgeGroup::default()
                };
                track = [EMPTY; 2];
            }
        }
        let count = (eg + 1) as usize;
        if contains_long && split_long_groups(&mut groups, &nedges, count) {
            has_singularities = true;
        }
        let valid = valid_len(&groups);
        for (g, group) in groups.iter_mut().take(valid).enumerate() {
            for &e in &group.edges {
                let flip = usize::from(vm[edge_list[nedges[e].old][1]] == i);
                nedges[e].links[flip] = Some((i, g));
            }
            if do_shell || (do_rim && !group.is_orig_closed) {
                group.new_vert = new_verts;
                new_verts += 1;
            }
        }
        vert_groups[i] = Some(groups);
    }

    Ok(Topo {
        sides,
        nedges,
        orig_edge_data,
        edge_lists,
        edge_len,
        vm,
        co,
        orig_edge_lengths,
        vert_groups,
        has_singularities,
        new_verts,
    })
}

fn is_elem(value: usize, a: usize, b: usize) -> bool {
    value == a || value == b
}

fn split_group(
    parent: &EdgeGroup,
    edges: Vec<usize>,
    is_even_split: bool,
    split: usize,
) -> EdgeGroup {
    EdgeGroup {
        valid: true,
        edges,
        is_orig_closed: parent.is_orig_closed,
        is_even_split,
        split,
        topo_group: parent.topo_group,
        ..EdgeGroup::default()
    }
}

/// Blender's split of self-intersecting edge groups (`contains_long_groups`): a group whose edges
/// repeat an original edge is cut at the repeats. Returns whether a singularity group was made.
fn split_long_groups(groups: &mut [EdgeGroup], nedges: &[NewEdge], count: usize) -> bool {
    let mut has_singularities = false;
    let mut splits = 0_usize;
    let mut add_index = 0_usize;
    for j in 0..count {
        let edges_len = groups[j + add_index].edges.len();
        if edges_len <= 3 {
            continue;
        }
        let g = groups[j + add_index].clone();
        let mut doubles = vec![false; edges_len];
        let mut has_doubles = false;
        for k in 0..edges_len {
            for l in k + 1..edges_len {
                if nedges[g.edges[k]].old == nedges[g.edges[l]].old {
                    doubles[k] = true;
                    doubles[l] = true;
                    has_doubles = true;
                }
            }
        }
        if !has_doubles {
            continue;
        }
        let prior_splits = splits;
        let prior_index = add_index;
        let mut unique_start: i64 = -1;
        let mut first_unique_end: i64 = -1;
        let mut last_split: i64 = -1;
        let mut first_split: i64 = -1;
        let mut first_even_split = false;
        let mut real_k = 0_usize;
        while real_k < edges_len
            || (g.is_orig_closed
                && (real_k as i64
                    <= (if first_unique_end == -1 {
                        0
                    } else {
                        first_unique_end
                    }) + edges_len as i64
                    || first_split != last_split))
        {
            let k = real_k % edges_len;
            if !doubles[k] {
                if first_unique_end != -1 && unique_start == -1 {
                    unique_start = real_k as i64;
                }
            } else if first_unique_end == -1 {
                first_unique_end = k as i64;
            } else if unique_start != -1 {
                let split = (unique_start as usize + real_k).div_ceil(2) % edges_len;
                let is_even_split = (unique_start as usize + real_k) & 1 == 1;
                if last_split != -1 {
                    // The first split overrides group `g` in place (no insert).
                    if prior_splits != splits {
                        shift_groups(groups, j + add_index + 1, j + add_index, count - j);
                        add_index += 1;
                    }
                    let last = last_split as usize;
                    let edges: Vec<usize> = if last > split {
                        g.edges[last..]
                            .iter()
                            .chain(g.edges[..split].iter())
                            .copied()
                            .collect()
                    } else {
                        g.edges[last..split].to_vec()
                    };
                    groups[j + add_index] = split_group(
                        &g,
                        edges,
                        is_even_split,
                        add_index - prior_index + 1 + usize::from(!g.is_orig_closed),
                    );
                    splits += 1;
                }
                last_split = split as i64;
                if first_split == -1 {
                    first_split = split as i64;
                    first_even_split = is_even_split;
                }
                unique_start = -1;
            }
            real_k += 1;
        }
        if first_split != -1 && !g.is_orig_closed {
            if prior_splits == splits {
                shift_groups(groups, j + add_index + 2, j + add_index + 1, count - j - 1);
            } else {
                shift_groups(
                    groups,
                    j + prior_index + 1,
                    j + prior_index,
                    count + add_index - (j + prior_index),
                );
                shift_groups(groups, j + add_index + 2, j + add_index + 1, count - j);
                add_index += 1;
            }
            let first = first_split as usize;
            groups[j + prior_index] =
                split_group(&g, g.edges[..first].to_vec(), first_even_split, 1);
            add_index += 1;
            splits += 1;
            let last = last_split as usize;
            groups[j + add_index] = split_group(
                &g,
                g.edges[last..].to_vec(),
                false,
                add_index - prior_index + 1,
            );
        }
        if first_unique_end != -1 && prior_splits == splits {
            has_singularities = true;
            groups[j + add_index].is_singularity = true;
        }
    }
    has_singularities
}

/// Blender `Solidify` face normal (`face_normals_true`): the normalized Newell normal, or zero for a
/// face without area.
fn face_normal_true(points: &[V3]) -> V3 {
    let mut normal: V3 = [0.0; 3];
    let count = points.len();
    for index in 0..count {
        let current = points[index];
        let previous = points[(index + count - 1) % count];
        normal[0] += (previous[1] - current[1]) * (previous[2] + current[2]);
        normal[1] += (previous[2] - current[2]) * (previous[0] + current[0]);
        normal[2] += (previous[0] - current[0]) * (previous[1] + current[1]);
    }
    normalize3(&mut normal);
    normal
}

/// Blender `BKE_id_defgroup_name_index(...) != -1` for a mesh vertex group.
fn has_vertex_group(mesh: &Mesh, name: &str) -> bool {
    !name.is_empty()
        && mesh
            .attributes
            .get("vertex_groups")
            .and_then(Value::as_object)
            .is_some_and(|groups| groups.contains_key(name))
}

/// Whether the mesh carries attribute `name`, as a top-level entry or a `blender_attributes` entry.
fn has_attribute(mesh: &Mesh, name: &str) -> bool {
    mesh.attributes.contains_key(name)
        || mesh
            .attributes
            .get("blender_attributes")
            .and_then(Value::as_array)
            .is_some_and(|entries| {
                entries
                    .iter()
                    .any(|entry| entry.get("name").and_then(Value::as_str) == Some(name))
            })
}

/// Storage-order float values of a `blender_attributes` entry or a top-level array.
fn positional_values(mesh: &Mesh, name: &str) -> Option<Vec<f32>> {
    mesh_attribute_values(mesh, name).map(|values| {
        values
            .iter()
            .map(|value| value.as_f64().unwrap_or(0.0) as f32)
            .collect()
    })
}

/// Storage-order edge values. An edge-id keyed map (`crease_edge`) reads absent keys as zero.
fn edge_values(mesh: &Mesh, name: &str) -> Option<Vec<f32>> {
    if let Some(values) = mesh
        .attributes
        .get(name)
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object)
    {
        return Some(
            mesh.edges
                .iter()
                .map(|edge| {
                    values
                        .get(&format!("e{}", edge.id))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0) as f32
                })
                .collect(),
        );
    }
    positional_values(mesh, name)
}

fn orig_value(values: &[f32], index: usize) -> f32 {
    values.get(index).copied().unwrap_or(0.0)
}

/// Blender's `short` range for `material_offset` and `material_offset_rim`.
fn clamp_i16(value: i64) -> i64 {
    value.clamp(i64::from(i16::MIN), i64::from(i16::MAX))
}

/// Output vertex of an edge endpoint. An endpoint without an output vertex is a broken fan.
fn link_vertex(
    vert_groups: &[Option<Vec<EdgeGroup>>],
    link: Option<(usize, usize)>,
) -> Result<usize> {
    link_group(vert_groups, link)
        .map(|group| group.new_vert)
        .filter(|&vert| vert != EMPTY)
        .ok_or_else(|| invariant("solidify edge endpoint has no output vertex"))
}

fn link_singular(vert_groups: &[Option<Vec<EdgeGroup>>], link: Option<(usize, usize)>) -> bool {
    link_group(vert_groups, link).is_some_and(|group| group.is_singularity)
}

fn out_edge(edges: &[[usize; 2]], new_edge: &[usize], id: usize) -> Result<[usize; 2]> {
    new_edge
        .get(id)
        .and_then(|&index| edges.get(index))
        .copied()
        .ok_or_else(|| invariant("solidify rim edge was not built"))
}

/// One output polygon: its vertices, the source corner behind each, its source face and material.
struct OutFace {
    verts: Vec<usize>,
    sources: Vec<usize>,
    face: usize,
    material: i64,
}

/// Sets output vertex `vertex_id` to weight 1 in the existing group `name` (`BKE_defvert_ensure_index`).
fn set_group_weight(mesh: &mut Mesh, name: &str, vertex_id: u32) {
    let groups = mesh
        .attributes
        .entry("vertex_groups".to_owned())
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(weights) = groups.as_object_mut().and_then(|groups| {
        groups
            .entry(name.to_owned())
            .or_insert_with(|| Value::Object(Map::new()))
            .as_object_mut()
    }) {
        weights.insert(format!("v{vertex_id}"), json!(1.0));
    }
}

fn remove_blender_attribute(mesh: &mut Mesh, name: &str) {
    mesh.attributes.remove(name);
    if let Some(Value::Array(entries)) = mesh.attributes.get_mut("blender_attributes") {
        entries.retain(|entry| entry.get("name").and_then(Value::as_str) != Some(name));
    }
}

/// `MOD_solidify_nonmanifold_modifyMesh`. Inputs that Blender returns unchanged come back as the
/// input mesh; the output is built from the per-vertex edge groups.
pub(super) fn solidify_nonmanifold(
    mesh: &Mesh,
    _modifier: &Modifier,
    params: &Params,
) -> Result<Mesh> {
    let verts_num = mesh.vertices.len();
    let faces_num = mesh.faces.len();
    if faces_num == 0 && verts_num != 0 {
        return Ok(mesh.clone());
    }
    let polys = polygons(mesh)?;
    let edge_list = edge_pairs(mesh)?;
    let edges_num = edge_list.len();
    let edge_of: HashMap<(usize, usize), usize> = edge_list
        .iter()
        .enumerate()
        .map(|(index, pair)| (edge_key(pair[0], pair[1]), index))
        .collect();
    let mut corners = Corners {
        face_start: Vec::with_capacity(faces_num),
        face_size: Vec::with_capacity(faces_num),
        corner_verts: Vec::new(),
        corner_edges: Vec::new(),
    };
    for (vertices, _) in &polys {
        corners.face_start.push(corners.corner_verts.len());
        corners.face_size.push(vertices.len());
        for index in 0..vertices.len() {
            let vert = vertices[index];
            let next = vertices[(index + 1) % vertices.len()];
            let edge = *edge_of.get(&edge_key(vert, next)).ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "solidify face edge is missing from the mesh",
                )
            })?;
            corners.corner_verts.push(vert);
            corners.corner_edges.push(edge);
        }
    }

    let orig_co: Vec<V3> = mesh
        .vertices
        .iter()
        .map(|vertex| to_v3(vertex.co))
        .collect();
    let mut face_nors: Vec<V3> = Vec::with_capacity(faces_num);
    for face in 0..faces_num {
        let start = corners.face_start[face];
        let points: Vec<V3> = corners.corner_verts[start..start + corners.face_size[face]]
            .iter()
            .map(|&vert| orig_co[vert])
            .collect();
        face_nors.push(face_normal_true(&points));
    }
    let mut null_faces = vec![false; faces_num];
    for face in 0..faces_num {
        if dot3(face_nors[face], face_nors[face]) < 0.5 {
            // A face without area takes the axis of its first edge that is least aligned with Y or Z.
            let edge = edge_list[corners.corner_edges[corners.face_start[face]]];
            let edgedir = sub3(orig_co[edge[1]], orig_co[edge[0]]);
            if edgedir[2].abs() < edgedir[1].abs() {
                face_nors[face][2] = 1.0;
            } else {
                face_nors[face][1] = 1.0;
            }
            null_faces[face] = true;
        }
    }

    let mat_nrs = i64::try_from(params.material_slot_count.max(1)).unwrap_or(i64::MAX);
    let mat_nr_max = mat_nrs - 1;
    let use_material_offsets = params.material_slot_count > 1;
    let mat_ofs = if use_material_offsets {
        clamp_i16(params.material_offset)
    } else {
        0
    };
    let mat_ofs_rim = if use_material_offsets {
        clamp_i16(params.material_offset_rim)
    } else {
        0
    };

    let offset = params.thickness as f32;
    let offset_fac = params.offset_fac as f32;
    let offset_clamp = params.offset_clamp as f32;
    let offset_fac_vg = params.offset_fac_vg as f32;
    let offset_fac_vg_inv = 1.0 - offset_fac_vg;
    let ofs_front = blender_midpoint(offset_fac, 1.0) * offset;
    let ofs_back = ofs_front - offset * offset_fac;
    let ofs_front_clamped = clamp_nonzero(ofs_front, 1.0e-5);
    let ofs_back_clamped = clamp_nonzero(ofs_back, 1.0e-5);
    let offset_abs = offset.abs() * offset_clamp;
    let do_flip = params.flip_normals == (offset > 0.0);
    let do_rim = params.rim;
    let do_shell = !(params.rim && params.rim_only);
    let do_clamp = offset_clamp != 0.0;
    let bevel_convex = params.bevel_convex as f32;
    let merge_threshold = params.merge_threshold as f32;

    // Thickness group weights, one per vertex. `use_flat_faces` scales each face by its vertices.
    // `dvert` may exist even when the requested thickness group has no matching index.
    let has_dvert = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .is_some_and(|groups| {
            groups.values().any(|weights| {
                weights
                    .as_object()
                    .is_some_and(|weights| !weights.is_empty())
            })
        });
    let vgroup: Option<Vec<f32>> = match (has_dvert, params.groups.thickness.as_ref()) {
        (true, Some(weights)) => Some(weights.iter().map(|&weight| weight as f32).collect()),
        _ => None,
    };
    let face_weight: Option<Vec<f32>> = match (&vgroup, params.flat_faces) {
        (Some(weights), true) => Some(
            (0..faces_num)
                .map(|face| {
                    let start = corners.face_start[face];
                    let mut scalar = 1.0_f32;
                    for &vert in &corners.corner_verts[start..start + corners.face_size[face]] {
                        let weight = weights[vert];
                        scalar = min_ff(
                            if params.invert_vertex_group {
                                1.0 - weight
                            } else {
                                weight
                            },
                            scalar,
                        );
                    }
                    offset_fac_vg + scalar * offset_fac_vg_inv
                })
                .collect(),
        ),
        _ => None,
    };
    let shell_present = has_vertex_group(mesh, &params.shell_group);
    let rim_present = has_vertex_group(mesh, &params.rim_group);

    let crease_src = edge_values(mesh, "crease_edge").filter(|values| !values.is_empty());
    let vertex_crease = positional_values(mesh, "crease_vert").filter(|values| !values.is_empty());
    let out_crease = crease_src.is_some() || vertex_crease.is_some();
    let bweight_src = edge_values(mesh, "bevel_weight_edge");
    let vert_bweight =
        positional_values(mesh, "bevel_weight_vert").filter(|values| !values.is_empty());
    let out_bweight =
        has_attribute(mesh, "bevel_weight_edge") || bevel_convex != 0.0 || vert_bweight.is_some();

    let mut topo = build_topology(
        &corners,
        &edge_list,
        &orig_co,
        &face_nors,
        do_shell,
        do_rim,
        merge_threshold,
    )?;

    // Offset position of every edge group.
    {
        let Topo {
            sides,
            nedges,
            vm,
            co,
            orig_edge_lengths,
            vert_groups,
            ..
        } = &mut topo;
        let geo = Geo {
            co: co.as_slice(),
            vm: vm.as_slice(),
            orig_edges: &edge_list,
            corner_verts: &corners.corner_verts,
            sides: sides.as_slice(),
            nedges: nedges.as_slice(),
            face_nors: &face_nors,
            null_faces: &null_faces,
            face_weight: face_weight.as_deref(),
            orig_edge_lengths: orig_edge_lengths.as_slice(),
            vgroup: vgroup.as_deref(),
            invert_vgroup: params.invert_vertex_group,
            offset_fac,
            offset_fac_vg,
            offset_fac_vg_inv,
            ofs_front: ofs_front_clamped,
            ofs_back: ofs_back_clamped,
            offset: offset_abs,
            do_clamp,
            do_angle_clamp: params.angle_clamp,
            thickness_mode: params.thickness_mode,
            boundary_mode: params.boundary_mode,
        };
        for (vertex, groups) in vert_groups.iter_mut().enumerate() {
            let Some(groups) = groups.as_mut() else {
                continue;
            };
            for group in 0..valid_len(groups) {
                let co = group_coord(&geo, vertex, &groups[group]);
                groups[group].co = co;
            }
        }
    }

    // Correction for adjacent one-sided singularity groups: each unordered pair gets one edge.
    let mut singularity_edges: Vec<[usize; 2]> = Vec::new();
    if topo.has_singularities {
        for i in 0..edges_num {
            let Some(data) = topo.orig_edge_data[i] else {
                continue;
            };
            if !(do_shell || topo.edge_len[i] == 1)
                || topo.nedges[topo.edge_lists[data][0]].old != i
            {
                continue;
            }
            for &edge in &topo.edge_lists[data] {
                let new = &topo.nedges[edge];
                if !link_singular(&topo.vert_groups, new.links[0])
                    || !link_singular(&topo.vert_groups, new.links[1])
                {
                    continue;
                }
                let pair = [
                    link_vertex(&topo.vert_groups, new.links[0])?,
                    link_vertex(&topo.vert_groups, new.links[1])?,
                ];
                if singularity_edges.iter().any(|prior| {
                    (prior[0] == pair[0] && prior[1] == pair[1])
                        || (prior[0] == pair[1] && prior[1] == pair[0])
                }) {
                    continue;
                }
                singularity_edges.push(pair);
            }
        }
    }
    let has_singularities = !singularity_edges.is_empty();
    let mut output_edges = singularity_edges.clone();
    let mut edge_values_out = vec![0.0_f32; output_edges.len()];
    let mut bevel_values_out = vec![0.0_f32; output_edges.len()];
    let mut edge_mapping = vec![EMPTY; topo.nedges.len()];
    // Original edge-derived edges come after the singularity edges.
    for i in 0..edges_num {
        let Some(data) = topo.orig_edge_data[i] else {
            continue;
        };
        if !(do_shell || topo.edge_len[i] == 1) || topo.nedges[topo.edge_lists[data][0]].old != i {
            continue;
        }
        for &edge in &topo.edge_lists[data] {
            if topo.nedges[edge].output_edge == EMPTY {
                continue;
            }
            let new = &topo.nedges[edge];
            let v1 = link_vertex(&topo.vert_groups, new.links[0])?;
            let v2 = link_vertex(&topo.vert_groups, new.links[1])?;
            let singular_pair = has_singularities
                && link_singular(&topo.vert_groups, new.links[0])
                && link_singular(&topo.vert_groups, new.links[1]);
            let insert = if singular_pair {
                singularity_edges
                    .iter()
                    .position(|pair| {
                        (pair[0] == v1 && pair[1] == v2) || (pair[0] == v2 && pair[1] == v1)
                    })
                    .ok_or_else(|| invariant("solidify singularity edge was not recorded"))?
            } else {
                let index = output_edges.len();
                output_edges.push([v1, v2]);
                edge_values_out.push(orig_value(crease_src.as_deref().unwrap_or(&[]), i));
                let mut bevel = orig_value(bweight_src.as_deref().unwrap_or(&[]), i);
                if bevel_convex != 0.0 && new.faces[1] != EMPTY {
                    let angle = new.angle as f64;
                    let convex = if angle > PI_F64 + f64::from(EPS) {
                        clamp_f(bevel_convex, 0.0, 1.0)
                    } else if angle < PI_F64 - f64::from(EPS) {
                        clamp_f(bevel_convex, -1.0, 0.0)
                    } else {
                        0.0
                    };
                    bevel = clamp_f(bevel + convex, 0.0, 1.0);
                }
                bevel_values_out.push(bevel);
                index
            };
            if singular_pair {
                edge_values_out[insert] = orig_value(crease_src.as_deref().unwrap_or(&[]), i);
                let mut bevel = orig_value(bweight_src.as_deref().unwrap_or(&[]), i);
                if bevel_convex != 0.0 && new.faces[1] != EMPTY {
                    let angle = new.angle as f64;
                    let convex = if angle > PI_F64 + f64::from(EPS) {
                        clamp_f(bevel_convex, 0.0, 1.0)
                    } else if angle < PI_F64 - f64::from(EPS) {
                        clamp_f(bevel_convex, -1.0, 0.0)
                    } else {
                        0.0
                    };
                    bevel = clamp_f(bevel + convex, 0.0, 1.0);
                }
                bevel_values_out[insert] = bevel;
            }
            edge_mapping[edge] = insert;
        }
    }

    // Positions follow Blender's `new_vert` ordering; unused input vertices disappear.
    let mut points = vec![DVec3::ZERO; topo.new_verts];
    let mut vertex_origins = vec![Vec::new(); topo.new_verts];
    for (source, groups) in topo.vert_groups.iter().enumerate() {
        let Some(groups) = groups else {
            continue;
        };
        for group in &groups[..valid_len(groups)] {
            if group.new_vert != EMPTY {
                points[group.new_vert] = to_dvec(group.co);
                vertex_origins[group.new_vert] = vec![(source, 1.0)];
            }
        }
    }

    // A boundary run connects the open edge groups around a vertex and closes a rim polygon.
    let mut out_faces: Vec<OutFace> = Vec::new();
    let mut boundary_edges: Vec<[usize; 2]> = Vec::new();
    let mut boundary_crease: Vec<f32> = Vec::new();
    let mut boundary_bevel: Vec<f32> = Vec::new();
    let mut rim_vertex_marks = HashSet::new();
    let mut shell_vertex_marks = HashSet::new();
    for vertex in 0..verts_num {
        let Some(groups) = topo.vert_groups[vertex].as_ref() else {
            continue;
        };
        let valid = valid_len(groups);
        let mut run_groups: Vec<usize> = Vec::new();
        let mut last_crease = 0.0_f32;
        let mut first_crease = 0.0_f32;
        let mut last_bevel = 0.0_f32;
        let mut first_bevel = 0.0_f32;
        let mut group_cursor = 0_usize;
        for g in 0..valid {
            let qualifies =
                (do_rim && !groups[g].is_orig_closed) || (do_shell && groups[g].split != 0);
            if qualifies {
                let edges = &groups[g].edges;
                let mut max_crease = 0.0_f32;
                let mut max_bevel = 0.0_f32;
                if edges.len() == 2 {
                    if out_crease && let Some(values) = crease_src.as_deref() {
                        max_crease = min_ff(
                            orig_value(values, topo.nedges[edges[0]].old),
                            orig_value(values, topo.nedges[edges[1]].old),
                        );
                    }
                } else {
                    for &edge in edges.iter().skip(1).take(edges.len().saturating_sub(2)) {
                        let old = topo.nedges[edge].old;
                        if out_crease && let Some(values) = crease_src.as_deref() {
                            max_crease = max_std(max_crease, orig_value(values, old));
                        }
                        if edge_mapping[edge] != EMPTY && out_bweight {
                            max_bevel = max_std(max_bevel, bevel_values_out[edge_mapping[edge]]);
                        }
                    }
                }
                let open_bevel = if let (Some(values), Some(first), Some(last)) =
                    (bweight_src.as_deref(), edges.first(), edges.last())
                {
                    min_ff(
                        orig_value(values, topo.nedges[*first].old),
                        orig_value(values, topo.nedges[*last].old),
                    )
                } else {
                    0.0
                };
                if open_bevel > 0.0 {
                    max_bevel = min_ff(open_bevel, max_bevel);
                } else if bevel_convex < 0.0 {
                    max_bevel = 0.0;
                }
                if let Some(&last) = run_groups.last() {
                    boundary_edges.push([groups[last].new_vert, groups[g].new_vert]);
                    boundary_crease.push(max_ff(
                        orig_value(vertex_crease.as_deref().unwrap_or(&[]), vertex),
                        min_ff(last_crease, max_crease),
                    ));
                    boundary_bevel.push(max_ff(
                        orig_value(vert_bweight.as_deref().unwrap_or(&[]), vertex),
                        min_ff(last_bevel, max_bevel),
                    ));
                } else {
                    first_crease = max_crease;
                    first_bevel = max_bevel;
                }
                run_groups.push(g);
                last_crease = max_crease;
                last_bevel = max_bevel;
            }
            if g + 1 == valid || groups[g].topo_group != groups[g + 1].topo_group {
                if run_groups.len() > 2 {
                    let first = run_groups[0];
                    let last = run_groups[run_groups.len() - 1];
                    boundary_edges.push([groups[last].new_vert, groups[first].new_vert]);
                    boundary_crease.push(max_ff(
                        orig_value(vertex_crease.as_deref().unwrap_or(&[]), vertex),
                        min_ff(last_crease, first_crease),
                    ));
                    boundary_bevel.push(max_ff(
                        orig_value(vert_bweight.as_deref().unwrap_or(&[]), vertex),
                        min_ff(last_bevel, first_bevel),
                    ));
                    let (material, source_face) = boundary_material(
                        groups,
                        group_cursor,
                        run_groups.len(),
                        &topo,
                        &polys,
                        mat_nrs,
                        do_rim,
                        do_shell,
                    );
                    let mut source_loops = Vec::with_capacity(run_groups.len());
                    let mut cursor = group_cursor;
                    while cursor < valid && source_loops.len() < run_groups.len() {
                        if (do_rim && !groups[cursor].is_orig_closed)
                            || (do_shell && groups[cursor].split != 0)
                        {
                            let side = topo.nedges[groups[cursor].edges[0]].faces[0];
                            let face = &topo.sides[side];
                            let loop_index = (0..face.size)
                                .map(|offset| face.start + offset)
                                .find(|&corner| topo.vm[corners.corner_verts[corner]] == vertex)
                                .ok_or_else(|| {
                                    invariant("solidify boundary face has no source corner")
                                })?;
                            source_loops.push(loop_index);
                        }
                        cursor += 1;
                    }
                    group_cursor = cursor;
                    if source_loops.len() != run_groups.len() {
                        return Err(invariant(
                            "solidify boundary face has incomplete corner data",
                        ));
                    }
                    let mut verts = Vec::with_capacity(run_groups.len());
                    let mut sources = Vec::with_capacity(run_groups.len());
                    if do_flip {
                        // `edges[edge_index-k][1]` and `loops_data[j-k]` in Blender.
                        verts.push(groups[first].new_vert);
                        sources.push(source_loops[source_loops.len() - 1]);
                        for index in (1..run_groups.len()).rev() {
                            verts.push(groups[run_groups[index]].new_vert);
                            sources.push(source_loops[index - 1]);
                        }
                    } else {
                        for (&group, &source) in run_groups.iter().zip(&source_loops) {
                            verts.push(groups[group].new_vert);
                            sources.push(source);
                        }
                    }
                    let material = if groups[g].is_orig_closed || !do_rim {
                        material
                    } else {
                        material + mat_ofs_rim
                    };
                    out_faces.push(OutFace {
                        verts,
                        sources,
                        face: source_face,
                        material: material.clamp(0, mat_nr_max),
                    });
                }
                run_groups.clear();
                last_crease = 0.0;
                first_crease = 0.0;
                last_bevel = 0.0;
                first_bevel = 0.0;
            }
        }
    }

    // Boundary edges follow the original-edge-derived edges in Blender's edge domain.
    for (index, edge) in boundary_edges.into_iter().enumerate() {
        output_edges.push(edge);
        edge_values_out.push(boundary_crease[index]);
        bevel_values_out.push(boundary_bevel[index]);
    }

    // Rim faces around original boundary edges.
    if do_rim {
        for i in 0..edges_num {
            if topo.edge_len[i] != 1 {
                continue;
            }
            let Some(data) = topo.orig_edge_data[i] else {
                continue;
            };
            let list = &topo.edge_lists[data];
            if list.len() < 2 || topo.nedges[list[0]].old != i {
                continue;
            }
            let edge1 = list[0];
            let edge2 = list[1];
            let e1 = &topo.nedges[edge1];
            let e2 = &topo.nedges[edge2];
            let v1_singularity = link_singular(&topo.vert_groups, e1.links[0])
                && link_singular(&topo.vert_groups, e2.links[0]);
            let v2_singularity = link_singular(&topo.vert_groups, e1.links[1])
                && link_singular(&topo.vert_groups, e2.links[1]);
            if v1_singularity && v2_singularity {
                continue;
            }
            let face_side = e1.faces[0];
            let source_face = topo.sides[face_side].face;
            let source_start = corners.face_start[source_face];
            let source_size = corners.face_size[source_face];
            let old_v1 = topo.vm[edge_list[e1.old][0]];
            let old_v2 = topo.vm[edge_list[e1.old][1]];
            let mut loop1 = None;
            let mut loop2 = None;
            for corner in source_start..source_start + source_size {
                let vertex = topo.vm[corners.corner_verts[corner]];
                if vertex == old_v1 {
                    loop1 = Some(corner);
                } else if vertex == old_v2 {
                    loop2 = Some(corner);
                }
            }
            let loop1 =
                loop1.ok_or_else(|| invariant("solidify rim face lacks first source corner"))?;
            let loop2 =
                loop2.ok_or_else(|| invariant("solidify rim face lacks second source corner"))?;
            let edge1_pair = out_edge(&output_edges, &edge_mapping, edge1)?;
            let edge2_pair = out_edge(&output_edges, &edge_mapping, edge2)?;
            let mut verts = Vec::with_capacity(4);
            let mut sources = Vec::with_capacity(4);
            if do_flip {
                if !v1_singularity {
                    verts.push(edge1_pair[0]);
                    sources.push(loop1);
                    rim_vertex_marks.insert(edge1_pair[0]);
                }
                verts.push(edge2_pair[0]);
                sources.push(loop1);
                rim_vertex_marks.insert(edge2_pair[0]);
                if !v2_singularity {
                    verts.push(edge2_pair[1]);
                    sources.push(loop2);
                    rim_vertex_marks.insert(edge2_pair[1]);
                }
                verts.push(edge1_pair[1]);
                sources.push(loop2);
                rim_vertex_marks.insert(edge1_pair[1]);
            } else {
                verts.push(edge1_pair[0]);
                sources.push(loop1);
                rim_vertex_marks.insert(edge1_pair[0]);
                if !v2_singularity {
                    verts.push(edge1_pair[1]);
                    sources.push(loop2);
                    rim_vertex_marks.insert(edge1_pair[1]);
                }
                verts.push(edge2_pair[1]);
                sources.push(loop2);
                rim_vertex_marks.insert(edge2_pair[1]);
                if !v1_singularity {
                    verts.push(edge2_pair[0]);
                    sources.push(loop1);
                    rim_vertex_marks.insert(edge2_pair[0]);
                }
            }
            out_faces.push(OutFace {
                verts,
                sources,
                face: source_face,
                material: (i64::from(polys[source_face].1) + mat_ofs_rim).clamp(0, mat_nr_max),
            });
        }
    }

    // Shell faces are the two oriented sides of every input face.
    if do_shell {
        for side_id in 0..faces_num * 2 {
            let side = &topo.sides[side_id];
            let mut totloop = side.size;
            while totloop > 0
                && (side.links[totloop - 1] == EMPTY
                    || topo.nedges[side.links[totloop - 1]].output_edge == EMPTY)
            {
                totloop -= 1;
            }
            if totloop == 0 {
                continue;
            }
            let mut face_verts = Vec::with_capacity(totloop * 2);
            let mut face_loops = Vec::with_capacity(totloop * 2);
            let mut valid_edges = 0_usize;
            for j in 0..totloop {
                let edge = side.links[j];
                if edge == EMPTY || topo.nedges[edge].output_edge == EMPTY {
                    continue;
                }
                valid_edges += 1;
                let old_edge = topo.nedges[edge].old;
                let current_vert = topo.vm[corners.corner_verts[side.start + j]];
                let flip = usize::from(topo.vm[edge_list[old_edge][1]] == current_vert);
                let new_v1 = link_vertex(&topo.vert_groups, topo.nedges[edge].links[flip])?;
                let new_v2 = link_vertex(&topo.vert_groups, topo.nedges[edge].links[1 - flip])?;
                if face_verts.last().is_none_or(|&last| last != new_v1) {
                    face_verts.push(new_v1);
                    face_loops.push(side.start + j);
                }
                if j < totloop - 1 || face_verts.first().is_none_or(|&first| first != new_v2) {
                    face_verts.push(new_v2);
                    face_loops.push(side.start + (j + 1) % totloop);
                }
            }
            if face_verts.len() <= 2 || valid_edges <= 2 {
                continue;
            }
            let mut verts = Vec::with_capacity(face_verts.len());
            let mut sources = Vec::with_capacity(face_verts.len());
            if side.reversed == do_flip {
                let mut index = face_verts.len() - 1;
                for next in 0..face_verts.len() {
                    verts.push(face_verts[index]);
                    sources.push(face_loops[index]);
                    index = next;
                }
            } else {
                shell_vertex_marks.extend(face_verts.iter().copied());
                for index in (0..face_verts.len()).rev() {
                    verts.push(face_verts[index]);
                    sources.push(face_loops[index]);
                }
            }
            let offset = if side.reversed == do_flip { 0 } else { mat_ofs };
            out_faces.push(OutFace {
                verts,
                sources,
                face: side.face,
                material: (i64::from(polys[side.face].1) + offset).clamp(0, mat_nr_max),
            });
        }
    }

    let mut covered = HashSet::new();
    for face in &out_faces {
        for index in 0..face.verts.len() {
            covered.insert(edge_key(
                face.verts[index],
                face.verts[(index + 1) % face.verts.len()],
            ));
        }
    }
    let loose: Vec<[usize; 2]> = output_edges
        .iter()
        .filter(|edge| !covered.contains(&edge_key(edge[0], edge[1])))
        .copied()
        .collect();
    let corner_sources: Vec<Option<usize>> = out_faces
        .iter()
        .flat_map(|face| face.sources.iter().copied().map(Some))
        .collect();
    let origin = Origin {
        vertices: vertex_origins,
        faces: out_faces.iter().map(|face| Some(face.face)).collect(),
        corners: Some(
            corner_sources
                .iter()
                .map(|source| source.map(|corner| vec![(corner, 1.0)]).unwrap_or_default())
                .collect(),
        ),
    };
    let output_polygons = out_faces
        .iter()
        .map(|face| {
            (
                face.verts.clone(),
                u32::try_from(face.material).unwrap_or(0),
            )
        })
        .collect();
    let mut result = build_mesh(points, output_polygons, loose, &origin, mesh)?;

    // Blender removes vertex creases if a rim is enabled; other point attributes are remapped.
    if do_rim && vertex_crease.is_some() {
        remove_blender_attribute(&mut result, "crease_vert");
    }
    if rim_present {
        for &vertex in &rim_vertex_marks {
            if let Some(id) = result.vertices.get(vertex).map(|item| item.id) {
                set_group_weight(&mut result, &params.rim_group, id);
            }
        }
    }
    if shell_present {
        for &vertex in &shell_vertex_marks {
            if let Some(id) = result.vertices.get(vertex).map(|item| item.id) {
                set_group_weight(&mut result, &params.shell_group, id);
            }
        }
    }

    // `build_mesh` derives edge order from polygons, so transfer edge attributes by vertex pair.
    let mut edge_data: HashMap<(usize, usize), (f32, f32)> = HashMap::new();
    for (index, edge) in output_edges.iter().enumerate() {
        edge_data.entry(edge_key(edge[0], edge[1])).or_insert((
            edge_values_out.get(index).copied().unwrap_or(0.0),
            bevel_values_out.get(index).copied().unwrap_or(0.0),
        ));
    }
    let output_indices = id_indices(&result);
    let mut crease_values = Map::new();
    let mut bevel_values = Map::new();
    for edge in &result.edges {
        let first = output_indices
            .get(&edge.vertices[0])
            .copied()
            .unwrap_or(EMPTY);
        let second = output_indices
            .get(&edge.vertices[1])
            .copied()
            .unwrap_or(EMPTY);
        let (crease, bevel) = edge_data
            .get(&edge_key(first, second))
            .copied()
            .unwrap_or((0.0, 0.0));
        crease_values.insert(format!("e{}", edge.id), json!(f64::from(crease)));
        bevel_values.insert(format!("e{}", edge.id), json!(f64::from(bevel)));
    }
    if out_crease {
        remove_blender_attribute(&mut result, "crease_edge");
        result.attributes.insert(
            "crease_edge".to_owned(),
            json!({"domain": "edge", "type": "float", "values": crease_values}),
        );
    }
    if out_bweight {
        remove_blender_attribute(&mut result, "bevel_weight_edge");
        result.attributes.insert(
            "bevel_weight_edge".to_owned(),
            json!({"domain": "edge", "type": "float", "values": bevel_values}),
        );
    }
    Ok(result)
}

fn boundary_material(
    groups: &[EdgeGroup],
    start: usize,
    count_groups: usize,
    topo: &Topo,
    polys: &[Polygon],
    material_count: i64,
    do_rim: bool,
    do_shell: bool,
) -> (i64, usize) {
    let qualifies =
        |group: &EdgeGroup| (do_rim && !group.is_orig_closed) || (do_shell && group.split != 0);
    let mut most_material = 0_i64;
    let mut most_face = 0_usize;
    let mut most_count = 0_usize;
    for material in 0..material_count {
        let mut count = 0_usize;
        let mut face = 0_usize;
        let mut remaining = count_groups;
        for group in groups.iter().skip(start) {
            if !qualifies(group) {
                continue;
            }
            let edges = &group.edges;
            if let Some(&first_edge) = edges.first() {
                let index = topo.sides[topo.nedges[first_edge].faces[0]].face;
                if i64::from(polys[index].1) == material {
                    face = index;
                    count += 1;
                }
            }
            if let Some(&last_edge) = edges.last() {
                let new = &topo.nedges[last_edge];
                let side = if new.faces[1] == EMPTY {
                    new.faces[0]
                } else {
                    new.faces[1]
                };
                let index = topo.sides[side].face;
                if i64::from(polys[index].1) == material {
                    face = index;
                    count += 1;
                }
            }
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }
        if count > most_count {
            most_material = material;
            most_face = face;
            most_count = count;
        }
    }
    (most_material, most_face)
}
