// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from OpenSubdiv (https://graphics.pixar.com/opensubdiv) far/catmarkPatchBuilder.cpp (GregoryConverter, CatmarkLimits), far/patchBuilder.cpp (identifyManifoldCornerSpan, GetIrregularPatchCornerSpans, assembleIrregularSourcePatch), Apache-2.0.

//! Gregory endcap control points for a leaf quad, built from the level's vertices. Each row is a
//! sparse combination of level vertices. A corner's ring is either closed (interior) or an open span
//! bounded by singular edges (boundaries and infinitely sharp creases), following `OpenSubdiv`'s
//! corner spans.

use std::f64::consts::PI;

use super::level::{Level, RULE_DART};
use super::patch::{Row, combine, tangent_coefficient};

/// Edges that bound a corner span: boundary, non-manifold and infinitely sharp edges.
pub(super) fn is_singular_edge(level: &Level, edge: usize) -> bool {
    let tag = level.etags[edge];
    tag.boundary || tag.non_manifold || tag.inf_sharp
}

/// The faces around a leaf corner that form its span, in order from the face at the corner's
/// outgoing side to the face at its incoming side. `leaf_index` is the leaf's index among them.
pub(super) struct Span {
    pub(super) faces: Vec<usize>,
    pub(super) leaf_index: usize,
    /// The span is bounded by singular edges (a boundary or sharp crease), not closed.
    pub(super) open: bool,
}

/// The other edge of `face` at `vertex`, given one of its two edges at that vertex.
fn other_edge_at(level: &Level, face: usize, vertex: usize, edge: usize) -> usize {
    let corner = level.corner_of(face, vertex);
    let n = level.faces[face].len();
    let outgoing = level.face_edges[face][corner];
    let incoming = level.face_edges[face][(corner + n - 1) % n];
    if edge == outgoing { incoming } else { outgoing }
}

/// Walks from `face` around `vertex` across non-singular edges in one direction. Returns the faces
/// passed (excluding `face`) and whether the walk came back to `face` (a closed ring).
fn walk(level: &Level, face: usize, vertex: usize, start_edge: usize) -> (Vec<usize>, bool) {
    let mut passed = Vec::new();
    let mut current = face;
    let mut edge = start_edge;
    loop {
        if is_singular_edge(level, edge) {
            return (passed, false);
        }
        let Some(next) = level.other_face(edge, current) else {
            return (passed, false);
        };
        if next == face {
            return (passed, true);
        }
        if passed.len() > level.faces.len() {
            return (passed, false);
        }
        passed.push(next);
        edge = other_edge_at(level, next, vertex, edge);
        current = next;
    }
}

/// The span of `face` at `vertex` (`OpenSubdiv` `identifyManifoldCornerSpan`).
pub(super) fn corner_span(level: &Level, face: usize, vertex: usize) -> Span {
    let corner = level.corner_of(face, vertex);
    let n = level.faces[face].len();
    let outgoing = level.face_edges[face][corner];
    let incoming = level.face_edges[face][(corner + n - 1) % n];
    let (toward_incoming, closed) = walk(level, face, vertex, incoming);
    if closed {
        let mut faces = vec![face];
        faces.extend(toward_incoming);
        return Span {
            faces,
            leaf_index: 0,
            open: false,
        };
    }
    let (toward_outgoing, _) = walk(level, face, vertex, outgoing);
    let mut faces: Vec<usize> = toward_outgoing.iter().rev().copied().collect();
    let face_in_span = faces.len();
    faces.push(face);
    faces.extend(toward_incoming);
    Span {
        faces,
        leaf_index: face_in_span,
        open: true,
    }
}

/// One corner of a leaf quad: its vertex, its ring, and its endcap class.
pub(super) struct Corner {
    pub(super) point: usize,
    /// `ring[2i]` is the far end of the spoke `i`, `ring[2i+1]` the diagonal vertex of face `i`.
    /// Open rings end with the far end of the last boundary spoke.
    pub(super) ring: Vec<usize>,
    /// Number of faces in the span (`numFaces`).
    pub(super) num_faces: usize,
    /// Number of spokes around the corner: faces for a closed ring, faces plus one for an open span.
    pub(super) valence: usize,
    /// Index of the leaf among the span's faces (`faceInRing`).
    pub(super) face_in_ring: usize,
    pub(super) boundary: bool,
    pub(super) sharp: bool,
    /// Regular corners (valence 4 closed, or two faces on a boundary) use the regular stencils.
    pub(super) regular: bool,
    /// Cosine of the corner's face angle.
    pub(super) cos_face_angle: f64,
}

/// Builds the endcap description of the leaf's corner `vertex` (`OpenSubdiv`
/// `assembleIrregularSourcePatch` and `GetIrregularPatchCornerSpans`).
pub(super) fn leaf_corner(level: &Level, face: usize, vertex: usize) -> Corner {
    let tag = level.vtags[vertex];
    let span = corner_span(level, face, vertex);
    let k = span.faces.len();
    // Spans are only identified where infinitely sharp edges can cut the ring; otherwise the whole
    // ring is the span.
    let test_inf_sharp_edges = tag.inf_sharp_edges && tag.rule != RULE_DART;
    let (num_faces, face_in_ring, boundary, span_faces) =
        if test_inf_sharp_edges || tag.non_manifold {
            (k, span.leaf_index, span.open, span.faces.clone())
        } else {
            let ring = fan(level, face, vertex);
            let index = ring.iter().position(|&f| f == face).unwrap_or(0);
            (ring.len(), index, tag.boundary, ring)
        };
    let valence = num_faces + usize::from(boundary);
    let mut sharp = if tag.corner {
        true
    } else if tag.non_manifold {
        tag.inf_sharp
    } else if test_inf_sharp_edges {
        !tag.inf_sharp_crease
    } else {
        tag.inf_sharp
    };
    // Legacy behaviour: a smooth boundary corner with a single face is sharp.
    if !sharp && tag.xordinary && tag.boundary && !tag.inf_sharp && !tag.non_manifold {
        sharp = num_faces == 1;
    }
    let regular = ((num_faces << usize::from(boundary)) == 4) && !sharp;
    let face_angle = if regular {
        PI / 2.0
    } else if boundary {
        PI / num_faces as f64
    } else {
        2.0 * PI / num_faces as f64
    };
    Corner {
        point: vertex,
        ring: ring_points(level, vertex, &span_faces, boundary),
        num_faces,
        valence,
        face_in_ring,
        boundary,
        sharp,
        regular,
        cos_face_angle: if regular { 0.0 } else { face_angle.cos() },
    }
}

/// Face-varying corner endcap. `OpenSubdiv`'s non-manifold span walk uses the face-edge index after
/// crossing an edge, which is observably different from a manifold fan walk on inconsistent winding.
pub(super) fn leaf_corner_fvar(level: &Level, face: usize, vertex: usize) -> Corner {
    if !level.vtags[vertex].non_manifold {
        return leaf_corner(level, face, vertex);
    }
    let tag = level.vtags[vertex];
    let span = non_manifold_corner_span(level, face, vertex);
    let num_faces = span.faces.len();
    let boundary = span.open;
    let valence = num_faces + usize::from(boundary);
    let sharp = tag.corner || tag.inf_sharp;
    let regular = ((num_faces << usize::from(boundary)) == 4) && !sharp;
    let face_angle = if regular {
        PI / 2.0
    } else if boundary {
        PI / num_faces as f64
    } else {
        2.0 * PI / num_faces as f64
    };
    Corner {
        point: vertex,
        ring: non_manifold_ring_points(level, &span.faces, &span.corners, boundary),
        num_faces,
        valence,
        face_in_ring: span.face_in_span,
        boundary,
        sharp,
        regular,
        cos_face_angle: if regular { 0.0 } else { face_angle.cos() },
    }
}

struct NonManifoldSpan {
    faces: Vec<usize>,
    /// The local face-edge indices used by `OpenSubdiv`'s non-manifold ring traversal.
    corners: Vec<usize>,
    face_in_span: usize,
    open: bool,
}

/// `OpenSubdiv`'s `identifyNonManifoldCornerSpan`, including its edge-index traversal for winding
/// configurations that cannot use the manifold vertex-face ordering.
#[expect(
    clippy::unwrap_used,
    reason = "a non-manifold corner span walks a face that has at least one corner"
)]
fn non_manifold_corner_span(level: &Level, face: usize, vertex: usize) -> NonManifoldSpan {
    let corner = level.corner_of(face, vertex);
    let n = level.faces[face].len();
    let leading_start = level.face_edges[face][corner];
    let trailing_start = level.face_edges[face][(corner + n - 1) % n];
    let mut num_faces = 1;
    let mut face_in_span = 0;
    let mut start_face = face;
    let mut start_corner = corner;
    let mut leading_face = face;
    let mut leading_edge = leading_start;
    let mut periodic = false;
    let mut steps = 0;
    while !is_singular_edge(level, leading_edge) {
        steps += 1;
        if steps > level.faces.len() {
            break;
        }
        num_faces += 1;
        face_in_span += 1;
        let Some(next_face) = level.other_face(leading_edge, leading_face) else {
            break;
        };
        let next_edges = &level.face_edges[next_face];
        let Some(edge_corner) = next_edges.iter().position(|&edge| edge == leading_edge) else {
            break;
        };
        start_face = next_face;
        start_corner = (edge_corner + 1) % next_edges.len();
        leading_face = next_face;
        leading_edge = next_edges[start_corner];
        if leading_edge == trailing_start {
            periodic = !is_singular_edge(level, leading_edge);
            break;
        }
    }
    let mut trailing_face = face;
    let mut trailing_edge = trailing_start;
    if trailing_edge != leading_edge {
        let mut steps = 0;
        while !is_singular_edge(level, trailing_edge) {
            steps += 1;
            if steps > level.faces.len() {
                break;
            }
            num_faces += 1;
            let Some(next_face) = level.other_face(trailing_edge, trailing_face) else {
                break;
            };
            let next_edges = &level.face_edges[next_face];
            let Some(edge_corner) = next_edges.iter().position(|&edge| edge == trailing_edge)
            else {
                break;
            };
            trailing_face = next_face;
            trailing_edge = next_edges[(edge_corner + next_edges.len() - 1) % next_edges.len()];
            if trailing_edge == leading_start {
                periodic = !is_singular_edge(level, trailing_edge);
                break;
            }
        }
    }
    let mut faces = vec![start_face];
    let mut corners = vec![start_corner];
    while faces.len() < num_faces {
        let current = *faces.last().unwrap();
        let current_corner = *corners.last().unwrap();
        let current_edges = &level.face_edges[current];
        let incoming =
            current_edges[(current_corner + current_edges.len() - 1) % current_edges.len()];
        let Some(next_face) = level.other_face(incoming, current) else {
            break;
        };
        if next_face == start_face {
            break;
        }
        let next_edges = &level.face_edges[next_face];
        let Some(next_corner) = next_edges.iter().position(|&edge| edge == incoming) else {
            break;
        };
        faces.push(next_face);
        corners.push(next_corner);
    }
    NonManifoldSpan {
        face_in_span: face_in_span.min(faces.len().saturating_sub(1)),
        faces,
        corners,
        open: !periodic,
    }
}

fn non_manifold_ring_points(
    level: &Level,
    faces: &[usize],
    corners: &[usize],
    open: bool,
) -> Vec<usize> {
    let mut ring = Vec::with_capacity(2 * faces.len() + usize::from(open));
    for (i, (&face, &corner)) in faces.iter().zip(corners).enumerate() {
        let points = &level.faces[face];
        let n = points.len();
        if n == 4 {
            ring.push(points[(corner + 1) % n]);
            ring.push(points[(corner + 2) % n]);
        } else {
            ring.push(points[(corner + 1) % n]);
        }
        if open && i + 1 == faces.len() {
            ring.push(points[(corner + n - 1) % n]);
        }
    }
    ring
}

/// The full fan of faces around `vertex`, starting at `face` and walking across the incoming edge of
/// each face. Singular edges do not end the fan.
pub(super) fn fan(level: &Level, face: usize, vertex: usize) -> Vec<usize> {
    let corner = level.corner_of(face, vertex);
    let n = level.faces[face].len();
    let mut faces = vec![face];
    let mut current = face;
    let mut edge = level.face_edges[face][(corner + n - 1) % n];
    while let Some(next) = level.other_face(edge, current) {
        if next == face || faces.len() > level.faces.len() {
            break;
        }
        faces.push(next);
        edge = other_edge_at(level, next, vertex, edge);
        current = next;
    }
    faces
}

/// Ring points: far vertices of the spokes alternate with the diagonal vertices of the faces.
fn ring_points(level: &Level, vertex: usize, faces: &[usize], open: bool) -> Vec<usize> {
    let mut ring = Vec::with_capacity(2 * faces.len() + usize::from(open));
    let k = faces.len();
    for &face in faces {
        let corner = level.corner_of(face, vertex);
        let n = level.faces[face].len();
        let outgoing = level.face_edges[face][corner];
        // Spoke i is the outgoing edge of face i; the closed ring's spoke before face 0 is its
        // incoming edge, which is the outgoing edge of face k - 1.
        let spoke = outgoing;
        let far = level.far_vertex(spoke, vertex);
        ring.push(far);
        ring.push(level.faces[face][(corner + 2) % n]);
    }
    if open {
        // The last boundary spoke: the incoming edge of the last face.
        let last = faces[k - 1];
        let corner = level.corner_of(last, vertex);
        let n = level.faces[last].len();
        let incoming = level.face_edges[last][(corner + n - 1) % n];
        ring.push(level.far_vertex(incoming, vertex));
    }
    ring
}

/// Row of the given terms.
fn row(terms: &[(usize, f64)]) -> Row {
    terms.to_vec()
}

/// Weighted sum of a point's own corner and ring entries: `weights[0]` is the corner,
/// `weights[j]` the ring entry `j - 1`.
fn weighted_row(corner: &Corner, weights: &[f64]) -> Row {
    weights
        .iter()
        .enumerate()
        .filter(|(_, w)| **w != 0.0)
        .map(|(j, &w)| {
            (
                if j == 0 {
                    corner.point
                } else {
                    corner.ring[j - 1]
                },
                w,
            )
        })
        .collect()
}

/// Control points of a Gregory patch: `[P_c, ep_c, em_c, fp_c, fm_c]` for each corner `c` at index
/// `5 * c + k`, following `OpenSubdiv`'s `GregoryConverter::Convert`.
pub(super) fn control_points(corners: &[Corner; 4]) -> [Row; 20] {
    let edge_points: Vec<(Row, Row, Row)> = (0..4).map(|c| edge_points(corners, c)).collect();
    let mut out: [Row; 20] = std::array::from_fn(|_| Vec::new());
    for c in 0..4 {
        let next = (c + 1) % 4;
        let prev = (c + 3) % 4;
        let corner = &corners[c];
        let mut fp_regular = corner.regular && corners[next].regular;
        let mut fm_regular = corner.regular && corners[prev].regular;
        let mut fp_copied = false;
        let mut fm_copied = false;
        if corner.boundary {
            let ep_on_boundary = corner.face_in_ring == 0;
            let em_on_boundary = corner.face_in_ring + 1 == corner.num_faces;
            if corner.num_faces > 1 {
                if ep_on_boundary {
                    fp_regular = fm_regular;
                    fp_copied = !fp_regular;
                }
                if em_on_boundary {
                    fm_regular = fp_regular;
                    fm_copied = !fm_regular;
                }
            } else {
                fp_regular = true;
                fm_regular = true;
            }
        }
        let fp = if fp_regular {
            regular_face_point(corners, c)
        } else if fp_copied {
            Vec::new()
        } else {
            irregular_face_point(corners, &edge_points, c, next, true)
        };
        let fm = if fm_regular {
            regular_face_point(corners, c)
        } else if fm_copied {
            Vec::new()
        } else {
            irregular_face_point(corners, &edge_points, c, prev, false)
        };
        let fp = if fp_copied { fm.clone() } else { fp };
        let fm = if fm_copied { fp.clone() } else { fm };
        let (p, ep, em) = &edge_points[c];
        out[5 * c].clone_from(p);
        out[5 * c + 1].clone_from(ep);
        out[5 * c + 2].clone_from(em);
        out[5 * c + 3] = fp;
        out[5 * c + 4] = fm;
    }
    out
}

/// The corner point and the two edge points of corner `c`, by its endcap class.
fn edge_points(corners: &[Corner; 4], c: usize) -> (Row, Row, Row) {
    let corner = &corners[c];
    let next = corners[(c + 1) % 4].point;
    let prev = corners[(c + 3) % 4].point;
    let p_c = corner.point;
    if corner.sharp {
        return (
            vec![(p_c, 1.0)],
            vec![(p_c, 2.0 / 3.0), (next, 1.0 / 3.0)],
            vec![(p_c, 2.0 / 3.0), (prev, 1.0 / 3.0)],
        );
    }
    if !corner.boundary {
        if corner.regular {
            regular_interior_edge_points(corner)
        } else {
            irregular_interior_edge_points(corner)
        }
    } else if corner.num_faces > 1 {
        if corner.regular {
            regular_boundary_edge_points(corner)
        } else {
            irregular_boundary_edge_points(corner)
        }
    } else {
        (
            vec![(p_c, 4.0 / 6.0), (next, 1.0 / 6.0), (prev, 1.0 / 6.0)],
            vec![(p_c, 2.0 / 3.0), (next, 1.0 / 3.0)],
            vec![(p_c, 2.0 / 3.0), (prev, 1.0 / 3.0)],
        )
    }
}

/// Regular interior corner (`OpenSubdiv` `assignRegularEdgePoints`, interior branch).
fn regular_interior_edge_points(corner: &Corner) -> (Row, Row, Row) {
    let r = &corner.ring;
    let p_c = corner.point;
    let p = row(&[
        (p_c, 4.0 / 9.0),
        (r[0], 1.0 / 9.0),
        (r[2], 1.0 / 9.0),
        (r[4], 1.0 / 9.0),
        (r[6], 1.0 / 9.0),
        (r[1], 1.0 / 36.0),
        (r[3], 1.0 / 36.0),
        (r[5], 1.0 / 36.0),
        (r[7], 1.0 / 36.0),
    ]);
    let ep = row(&[
        (p_c, 4.0 / 9.0),
        (r[0], 2.0 / 9.0),
        (r[2], 1.0 / 9.0),
        (r[6], 1.0 / 9.0),
        (r[1], 1.0 / 18.0),
        (r[7], 1.0 / 18.0),
    ]);
    let em = row(&[
        (p_c, 4.0 / 9.0),
        (r[2], 2.0 / 9.0),
        (r[0], 1.0 / 9.0),
        (r[4], 1.0 / 9.0),
        (r[1], 1.0 / 18.0),
        (r[3], 1.0 / 18.0),
    ]);
    (p, ep, em)
}

/// Regular boundary corner (two faces; `OpenSubdiv` `assignRegularEdgePoints`, boundary branch).
fn regular_boundary_edge_points(corner: &Corner) -> (Row, Row, Row) {
    let r = &corner.ring;
    let p_c = corner.point;
    let p = row(&[(p_c, 2.0 / 3.0), (r[0], 1.0 / 6.0), (r[4], 1.0 / 6.0)]);
    let boundary_edge = if corner.face_in_ring == 0 {
        row(&[(p_c, 2.0 / 3.0), (r[0], 1.0 / 3.0)])
    } else {
        row(&[(p_c, 2.0 / 3.0), (r[4], 1.0 / 3.0)])
    };
    let interior = row(&[
        (p_c, 4.0 / 9.0),
        (r[2], 2.0 / 9.0),
        (r[0], 1.0 / 9.0),
        (r[4], 1.0 / 9.0),
        (r[1], 1.0 / 18.0),
        (r[3], 1.0 / 18.0),
    ]);
    if corner.face_in_ring == 0 {
        (p, boundary_edge, interior)
    } else {
        (p, interior, boundary_edge)
    }
}

/// Irregular interior corner (`OpenSubdiv` `ComputeInteriorPointWeights`), with the leaf at ring index
/// `face_in_ring`.
fn irregular_interior_edge_points(corner: &Corner) -> (Row, Row, Row) {
    let n = corner.valence;
    let inv_n = 1.0 / n as f64;
    let inv_n5 = 1.0 / (n as f64 + 5.0);
    let p_coeff = inv_n * inv_n5;
    let tan_coeff = tangent_coefficient(n) * 0.5 * inv_n5;
    let face_angle = 2.0 * PI * inv_n;
    let width = 2 * n + 1;

    let mut p_w = vec![0.0; width];
    p_w[0] = n as f64 * inv_n5;
    let mut t_w = vec![0.0; width];
    for i in 0..n {
        p_w[1 + 2 * i] = 4.0 * p_coeff;
        p_w[2 + 2 * i] = p_coeff;
        let i_prev = (i + n - 1) % n;
        let i_next = (i + 1) % n;
        let c_i = tan_coeff * (face_angle * i as f64).cos();
        t_w[1 + 2 * i_prev] += 2.0 * c_i;
        t_w[2 + 2 * i_prev] += c_i;
        t_w[1 + 2 * i] += 4.0 * c_i;
        t_w[2 + 2 * i] += c_i;
        t_w[1 + 2 * i_next] += 2.0 * c_i;
    }

    let f = corner.face_in_ring;
    let ep_offset = 2 * ((n - f) % n);
    let em_offset = 2 * ((n + n - f - 1) % n);
    let rotated = |i: usize, offset: usize| -> usize { 1 + ((i - 1 + offset) % (2 * n)) };
    let mut ep_w = vec![0.0; width];
    let mut em_w = vec![0.0; width];
    ep_w[0] = p_w[0];
    em_w[0] = p_w[0];
    for i in 1..width {
        ep_w[i] = p_w[i] + t_w[rotated(i, ep_offset)];
        em_w[i] = p_w[i] + t_w[rotated(i, em_offset)];
    }
    (
        weighted_row(corner, &p_w),
        weighted_row(corner, &ep_w),
        weighted_row(corner, &em_w),
    )
}

/// Irregular boundary corner (`OpenSubdiv` `ComputeBoundaryPointWeights`).
fn irregular_boundary_edge_points(corner: &Corner) -> (Row, Row, Row) {
    let (p_w, ep_w, em_w) = boundary_point_weights(corner.valence, corner.face_in_ring);
    (
        weighted_row(corner, &p_w),
        weighted_row(corner, &ep_w),
        weighted_row(corner, &em_w),
    )
}

/// Weights of the boundary corner's point and its edge points, `2 * valence` wide (index 0 is the
/// corner, index `1 + j` the ring entry `j`).
fn boundary_point_weights(valence: usize, face_in_ring: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let num_faces = valence - 1;
    let face_angle = PI / num_faces as f64;
    let width = 2 * valence;
    let last = width - 1;
    let mut p = vec![0.0; width];
    p[0] = 4.0 / 6.0;
    p[1] = 1.0 / 6.0;
    p[last] = 1.0 / 6.0;
    let t_coeff_1 = 1.0 / 6.0;
    let t_coeff_last = -1.0 / 6.0;

    let k = num_faces as f64;
    let theta = face_angle;
    let c = theta.cos();
    let s = theta.sin();
    let div3kc = 1.0 / (3.0 * k + c);
    let gamma = -4.0 * s * div3kc;
    let alpha_0k = -((1.0 + 2.0 * c) * (1.0 + c).sqrt()) * div3kc / (1.0 - c).sqrt();
    let beta_0 = s * div3kc;
    let mut t = vec![0.0; width];
    t[0] = gamma / 3.0;
    t[1] = alpha_0k / 3.0;
    t[2] = beta_0 / 3.0;
    t[last] = alpha_0k / 3.0;
    for i in 1..valence - 1 {
        let sin_i = (theta * i as f64).sin();
        let sin_next = (theta * (i + 1) as f64).sin();
        let alpha = 4.0 * sin_i * div3kc;
        let beta = (sin_i + sin_next) * div3kc;
        t[1 + 2 * i] = alpha / 3.0;
        t[1 + 2 * i + 1] = beta / 3.0;
    }

    let ep = if face_in_ring == 0 {
        let mut ep = vec![0.0; width];
        ep[0] = 2.0 / 3.0;
        ep[1] = 1.0 / 3.0;
        ep
    } else {
        let angle = face_angle * face_in_ring as f64;
        let (cos_a, sin_a) = (angle.cos(), angle.sin());
        let mut ep: Vec<f64> = t.iter().map(|w| w * sin_a).collect();
        ep[0] += p[0];
        ep[1] += p[1] + t_coeff_1 * cos_a;
        ep[last] += p[last] + t_coeff_last * cos_a;
        ep
    };
    let em = if face_in_ring + 1 == num_faces {
        let mut em = vec![0.0; width];
        em[0] = 2.0 / 3.0;
        em[last] = 1.0 / 3.0;
        em
    } else {
        let angle = face_angle * ((face_in_ring + 1) % valence) as f64;
        let (cos_a, sin_a) = (angle.cos(), angle.sin());
        let mut em: Vec<f64> = t.iter().map(|w| w * sin_a).collect();
        em[0] += p[0];
        em[1] += p[1] + t_coeff_1 * cos_a;
        em[last] += p[last] + t_coeff_last * cos_a;
        em
    };
    (p, ep, em)
}

/// Regular face point (`OpenSubdiv` `assignRegularFacePoints`).
fn regular_face_point(corners: &[Corner; 4], c: usize) -> Row {
    row(&[
        (corners[c].point, 4.0 / 9.0),
        (corners[(c + 3) % 4].point, 2.0 / 9.0),
        (corners[(c + 1) % 4].point, 2.0 / 9.0),
        (corners[(c + 2) % 4].point, 1.0 / 9.0),
    ])
}

/// Irregular face point (`OpenSubdiv` `computeIrregularFacePoint`): the face beside corner `c` toward
/// `far`, shifted by the ring corrections of the spokes around the near corner.
fn irregular_face_point(
    corners: &[Corner; 4],
    edge_points: &[(Row, Row, Row)],
    c: usize,
    far: usize,
    plus: bool,
) -> Row {
    let near = &corners[c];
    let far_corner = &corners[far];
    let cos_near = near.cos_face_angle;
    let cos_far = far_corner.cos_face_angle;
    let p_coeff = cos_far / 3.0;
    let near_coeff = (3.0 - 2.0 * cos_near - cos_far) / 3.0;
    let far_coeff = 2.0 * cos_near / 3.0;
    let sign = if plus { 1.0 } else { -1.0 };
    let (near_row, far_row) = if plus {
        (&edge_points[c].1, &edge_points[far].2)
    } else {
        (&edge_points[c].2, &edge_points[far].1)
    };
    let valence = near.valence;
    let edge_in_ring = if plus {
        near.face_in_ring
    } else {
        (near.face_in_ring + 1) % valence
    };
    let edge_prev = (edge_in_ring + valence - 1) % valence;
    let edge_next = (edge_in_ring + 1) % valence;
    let mut terms = combine(&[
        (p_coeff, &edge_points[c].0),
        (near_coeff, near_row),
        (far_coeff, far_row),
    ]);
    let r = &near.ring;
    terms.push((r[2 * edge_prev], -sign / 9.0));
    terms.push((r[2 * edge_prev + 1], -sign / 18.0));
    terms.push((r[2 * edge_in_ring + 1], sign / 18.0));
    terms.push((r[2 * edge_next], sign / 9.0));
    terms
}
