// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from OpenSubdiv (https://graphics.pixar.com/opensubdiv) vtr/fvarLevel.cpp
// (completeTopologyFromFaceValues, gatherValueSpans) and vtr/level.cpp (orderVertexFacesAndEdges),
// Apache-2.0.

//! Face-varying base level of a UV layer. A value is one face-varying vertex: `OpenSubdiv`'s sibling
//! of a position vertex, which is one UV island around it. The base level's value sharpness follows
//! `OpenSubdiv`'s face-varying tags, and the later levels refine over the position refinement's quads
//! (`refine_values`), so values live on the same leaves as positions.

use std::collections::HashMap;

use glam::DVec2;

use super::super::subdivision_uv::{FvarLinear, ValueMap};
use super::{
    level::{
        Level, RULE_CORNER, RULE_CREASE, SHARPNESS_INFINITE, VTag, is_infinite, is_semi_sharp,
    },
    refine::{Children, Refined},
};

impl FvarLinear {
    /// Linear boundaries make every value on a boundary vertex a corner (`_hasLinearBoundaries`).
    fn linear_boundaries(self) -> bool {
        matches!(self, Self::All | Self::Boundaries)
    }

    /// Single-face values are sharp unless the channel is smooth everywhere (`fvarCornersAreSharp`).
    fn corners_sharp(self) -> bool {
        self != Self::SmoothAll
    }

    /// Values depend on the sharpness of their neighbours (`_hasDependentSharpness`).
    fn dependent(self) -> bool {
        matches!(self, Self::CornersPlus1 | Self::CornersPlus2)
    }

    /// One sharp value sharpens the other (`sharpenBothIfOneCorner`).
    fn sharpen_both_if_one_corner(self) -> bool {
        self == Self::CornersPlus2
    }

    /// Single-value darts are sharp (`sharpenDarts`).
    fn sharpen_darts(self) -> bool {
        matches!(self, Self::All | Self::Boundaries | Self::CornersPlus2)
    }
}

/// `OpenSubdiv`'s `FVar` `ValueTag`, before it is combined with the value-level `VTag`.
#[derive(Clone, Copy, Default)]
struct FVarTag {
    mismatch: bool,
    xordinary: bool,
    non_manifold: bool,
    crease: bool,
    semi_sharp: bool,
    inf_sharp_edges: bool,
    inf_irregular: bool,
}

impl FVarTag {
    /// `FVarLevel::ValueTag::combineWithLevelVTag`.
    fn combine_with_level(self, mut tag: VTag) -> VTag {
        if !self.mismatch {
            return tag;
        }
        tag.rule = if self.crease {
            RULE_CREASE
        } else {
            RULE_CORNER
        };
        if self.crease || self.semi_sharp {
            tag.inf_sharp = false;
            tag.inf_sharp_crease = true;
            tag.corner = false;
        } else {
            tag.inf_sharp = true;
            tag.inf_sharp_crease = false;
            tag.corner = !self.inf_irregular && !self.inf_sharp_edges;
        }
        tag.inf_sharp_edges = true;
        tag.inf_irregular = self.inf_irregular;
        tag.boundary = true;
        tag.xordinary = self.xordinary;
        tag.non_manifold |= self.non_manifold;
        tag
    }
}

/// The base level of a UV layer over the base faces. `geometry` is the base position level: its
/// vertex and edge sharpness are the geometric ones, which a value inherits unless the channel's
/// options sharpen it.
#[expect(
    clippy::expect_used,
    reason = "every value edge is a position edge, since value edges are built from the position topology"
)]
pub(super) fn base(
    geometry: &Level,
    linear: FvarLinear,
    preserve_corners: bool,
    input: &ValueMap,
) -> Refined<DVec2> {
    let (map, mut sharpness, mut non_manifold, fvar_tags, dependent_sources, linear_edges) =
        values(geometry, linear, preserve_corners, input);
    let mut level = Level::from_faces(map.corners.clone(), map.uvs.len());
    let geometric_edges: HashMap<(usize, usize), usize> = geometry
        .edges
        .iter()
        .enumerate()
        .map(|(edge, &[a, b])| ((a, b), edge))
        .collect();
    for (edge, &[a, b]) in level.edges.iter().enumerate() {
        let (p, q) = (map.vertex[a], map.vertex[b]);
        let geometric = geometric_edges
            .get(&(p.min(q), p.max(q)))
            .expect("every value edge is a position edge");
        level.edge_sharpness[edge] = geometry.edge_sharpness[*geometric];
    }
    if linear == FvarLinear::All {
        level.edge_sharpness.fill(SHARPNESS_INFINITE);
        sharpness.fill(SHARPNESS_INFINITE);
        non_manifold.fill(false);
    }
    level.vertex_sharpness = sharpness;
    // Boundary edges (one face in the channel) are infinitely sharp; the edges' own sharpness stays.
    level.derive_base_tags(false, &non_manifold);
    for edge in 0..level.edge_count() {
        let [a, b] = level.edges[edge];
        let position_edge = geometric_edges[&(
            map.vertex[a].min(map.vertex[b]),
            map.vertex[a].max(map.vertex[b]),
        )];
        level.etags[edge].non_manifold |= geometry.etags[position_edge].non_manifold
            || is_inconsistently_wound_edge(geometry, position_edge);
        level.etags[edge].linear = linear_edges[position_edge];
    }
    for (value, tag) in fvar_tags.into_iter().enumerate() {
        level.vtags[value] = tag.combine_with_level(geometry.vtags[map.vertex[value]]);
    }
    Refined {
        level,
        positions: map.uvs.clone(),
        face_children: Vec::new(),
        dependent_source: dependent_sources,
        children: Children::default(),
    }
}

/// One position vertex's faces and edges in `OpenSubdiv`'s fan order.
struct Fan {
    /// Faces in fan order. For a boundary vertex the first face is the one with the forward boundary
    /// edge.
    faces: Vec<usize>,
    /// Edges in fan order: `edges[i]` joins `faces[i - 1]` and `faces[i]`. A boundary vertex has one
    /// more edge than faces, with its two boundary edges at both ends.
    edges: Vec<usize>,
    boundary: bool,
}

/// `OpenSubdiv`'s `orderVertexFacesAndEdges`: walks the fan around `vertex` from its first face (an
/// interior vertex) or from its forward boundary edge. `None` when the fan cannot be ordered.
fn fan(geometry: &Level, vertex: usize) -> Option<Fan> {
    let incident_faces = &geometry.vertex_faces[vertex];
    let incident_edges = &geometry.vertex_edges[vertex];
    let (fcount, ecount) = (incident_faces.len(), incident_edges.len());
    if fcount == 0 || ecount < 2 || ecount > fcount + 1 {
        return None;
    }
    let boundary = ecount != fcount;
    let (mut face, mut edge, mut corner) = if boundary {
        let mut start = None;
        for &e in incident_edges {
            if let &[only] = geometry.edge_faces[e].as_slice() {
                let c = geometry.corner_of(only, vertex);
                start = Some((only, e, c));
                if e == geometry.face_edges[only][c] {
                    break;
                }
            }
        }
        start?
    } else {
        let f = incident_faces[0];
        let c = geometry.corner_of(f, vertex);
        (f, geometry.face_edges[f][c], c)
    };
    let first_edge = edge;
    let mut faces = vec![face];
    let mut edges = vec![edge];
    while edges.len() < ecount {
        let n = geometry.faces[face].len();
        let previous = if corner == 0 { n - 1 } else { corner - 1 };
        let next_edge = geometry.face_edges[face][previous];
        if next_edge == edge || next_edge == first_edge {
            return None;
        }
        edges.push(next_edge);
        if faces.len() < fcount {
            let around = &geometry.edge_faces[next_edge];
            match around.as_slice() {
                [] => return None,
                [only] if *only == face => return None,
                _ => {}
            }
            face = if around[0] == face {
                *around.get(1)?
            } else {
                around[0]
            };
            corner = geometry.face_edges[face]
                .iter()
                .position(|&e| e == next_edge)?;
            if geometry.faces[face][corner] != vertex {
                return None;
            }
            faces.push(face);
        }
        edge = next_edge;
    }
    Some(Fan {
        faces,
        edges,
        boundary,
    })
}

/// A position vertex's faces with their values, as `OpenSubdiv`'s sibling assignment sees them.
struct Vertex {
    fan: Fan,
    /// The vertex is non-manifold: its faces stay in face order and every value is a non-manifold
    /// corner.
    non_manifold: bool,
    /// Sibling (output value, in order) of each face of `fan.faces`.
    sibling: Vec<usize>,
    /// Input value of each sibling.
    siblings: Vec<usize>,
}

/// `OpenSubdiv`'s sibling assignment over one vertex's faces. A value's siblings are numbered in order
/// of first appearance; after two distinct values are known, a different value is appended without
/// checking the second one. That quirk makes a repeated second value a new sibling, so an alternating
/// fan `A B A B` has siblings `0 1 0 2`.
fn assign_siblings(values: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut unique = vec![values[0]];
    let mut sibling = vec![0; values.len()];
    for i in 1..values.len() {
        if values[i] == values[i - 1] {
            sibling[i] = sibling[i - 1];
            continue;
        }
        sibling[i] = unique.len();
        if unique.len() == 1 || (unique.len() == 2 && unique[0] != values[i]) {
            unique.push(values[i]);
        } else if let Some(found) = unique.iter().position(|&value| value == values[i]) {
            sibling[i] = found;
        } else {
            unique.push(values[i]);
        }
    }
    (unique, sibling)
}

/// Spans of one sibling around its vertex (`OpenSubdiv`'s `ValueSpan`).
#[derive(Clone, Copy, Default)]
struct Span {
    /// Faces carrying the value.
    size: usize,
    /// Discontinuous edges inside the value's faces: more than zero means the value is disjoint (a
    /// bowtie) and therefore non-manifold.
    disct: i64,
    /// Infinitely sharp geometric edges inside the value's faces.
    inf: usize,
    /// Semi-sharp geometric edges inside the value's faces.
    semi: usize,
}

/// `OpenSubdiv`'s `gatherValueSpans`, including the wrap-around bookkeeping for the first sibling.
fn value_spans(geometry: &Level, vertex: &Vertex, mismatch: &[bool], count: usize) -> Vec<Span> {
    let n = vertex.fan.faces.len();
    let sibling = &vertex.sibling;
    let edges = &vertex.fan.edges;
    let mut spans = vec![Span::default(); count];
    if vertex.non_manifold {
        for span in &mut spans {
            span.disct = 1;
        }
    } else if count == 1 && !vertex.fan.boundary {
        // A single interior value is disjoint when two of its edges are discontinuous (a dart).
        let mut size = 0;
        for &edge in edges {
            if mismatch[edge] {
                if size > 0 {
                    spans[0].disct = 1;
                    break;
                }
                size = n;
            } else if geometry.etags[edge].inf_sharp {
                spans[0].inf += 1;
            } else if geometry.etags[edge].semi_sharp {
                spans[0].semi += 1;
            }
        }
        spans[0].size = n;
    } else {
        spans[0].size = 1;
        if !vertex.fan.boundary && sibling[n - 1] == 0 {
            let edge = edges[0];
            if mismatch[edge] {
                spans[0].disct += 1;
            } else if geometry.etags[edge].inf_sharp {
                spans[0].inf += 1;
            } else if geometry.etags[edge].semi_sharp {
                spans[0].semi += 1;
            }
        }
        for i in 1..n {
            let s = sibling[i];
            if sibling[i] == sibling[i - 1] {
                let edge = edges[i];
                if mismatch[edge] {
                    spans[s].disct += 1;
                } else if geometry.etags[edge].inf_sharp {
                    spans[s].inf += 1;
                } else if geometry.etags[edge].semi_sharp {
                    spans[s].semi += 1;
                }
            } else if spans[s].size > 0 {
                // The value returns after another value: its span is disjoint.
                spans[s].disct += 1;
            }
            spans[s].size += 1;
        }
        // A span that wraps around from the last face to the first is one span, not two.
        if sibling[n - 1] == 0 && !vertex.fan.boundary {
            spans[0].disct -= 1;
        }
    }
    spans
}

/// Maximum semi-sharpness among the geometry vertex and semi-sharp edges in a span. A dependent
/// value stays a corner until every source feature has decayed, so its synthetic vertex sharpness is
/// the maximum, while its fractional transition weight is computed from the source sibling.
fn span_semi_sharpness(
    geometry: &Level,
    vertex: &Vertex,
    position: usize,
    sibling: usize,
) -> Option<f64> {
    let mut sharpness = 0.0_f64;
    let vertex_sharpness = geometry.vertex_sharpness[position];
    let mut found = false;
    if is_semi_sharp(vertex_sharpness) {
        sharpness = sharpness.max(vertex_sharpness);
        found = true;
    }
    for i in 1..vertex.fan.faces.len() {
        if vertex.sibling[i] == sibling && vertex.sibling[i - 1] == sibling {
            let edge_sharpness = geometry.edge_sharpness[vertex.fan.edges[i]];
            if is_semi_sharp(edge_sharpness) {
                sharpness = sharpness.max(edge_sharpness);
                found = true;
            }
        }
    }
    if !vertex.fan.boundary
        && vertex.sibling[0] == sibling
        && vertex.sibling[vertex.sibling.len() - 1] == sibling
    {
        let edge_sharpness = geometry.edge_sharpness[vertex.fan.edges[0]];
        if is_semi_sharp(edge_sharpness) {
            sharpness = sharpness.max(edge_sharpness);
            found = true;
        }
    }
    found.then_some(sharpness)
}

/// Per-channel output of `values`: the fvar values, their sharpness, tags, and per-edge and per-vertex flags.
type ValuesOutput = (
    ValueMap,
    Vec<f64>,
    Vec<bool>,
    Vec<FVarTag>,
    Vec<Option<usize>>,
    Vec<bool>,
);

/// Builds fvar values and their base tags. Position vertices whose values form one smooth UV island
/// keep one value, with the geometric sharpness.
fn values(
    geometry: &Level,
    linear: FvarLinear,
    preserve_corners: bool,
    input: &ValueMap,
) -> ValuesOutput {
    let vertex_count = geometry.vertex_count();
    let make_smooth_corners_sharp = !preserve_corners && linear.corners_sharp();
    let mut mismatch = vec![false; geometry.edge_count()];
    let mut mismatched_vertex = vec![false; vertex_count];
    let mut linear_edges = vec![false; geometry.edge_count()];
    let mut vertices: Vec<Option<Vertex>> = (0..vertex_count).map(|_| None).collect();
    #[expect(
        clippy::needless_range_loop,
        reason = "each position indexes several per-vertex arrays and the geometry at once"
    )]
    for position in 0..vertex_count {
        if geometry.vertex_faces[position].is_empty() {
            continue;
        }
        let tag = &geometry.vtags[position];
        let ordered = if tag.non_manifold {
            None
        } else {
            fan(geometry, position)
        };
        let non_manifold = ordered.is_none();
        let fan = ordered.unwrap_or_else(|| Fan {
            faces: geometry.vertex_faces[position].clone(),
            edges: geometry.vertex_edges[position].clone(),
            boundary: tag.boundary,
        });
        let input_values: Vec<usize> = fan
            .faces
            .iter()
            .map(|&face| input.corners[face][geometry.corner_of(face, position)])
            .collect();
        let (siblings, sibling) = assign_siblings(&input_values);
        let n = fan.faces.len();
        if non_manifold {
            // Each edge with several faces is discontinuous when a face disagrees with the first.
            for &edge in &geometry.vertex_edges[position] {
                let around = &geometry.edge_faces[edge];
                if around.len() < 2 {
                    continue;
                }
                let value_at =
                    |face: usize| input.corners[face][geometry.corner_of(face, position)];
                let first = value_at(around[0]);
                if around[1..].iter().any(|&face| value_at(face) != first) {
                    mismatch[edge] = true;
                    linear_edges[edge] |= linear.linear_boundaries();
                    mark_endpoints(geometry, edge, &mut mismatched_vertex);
                }
            }
        } else {
            let start = usize::from(fan.boundary);
            for i in start..n {
                let previous = if i == 0 { n - 1 } else { i - 1 };
                if input_values[i] != input_values[previous] {
                    let edge = fan.edges[i];
                    mismatch[edge] = true;
                    linear_edges[edge] |= linear.linear_boundaries();
                    mark_endpoints(geometry, edge, &mut mismatched_vertex);
                }
            }
        }
        vertices[position] = Some(Vertex {
            fan,
            non_manifold,
            sibling,
            siblings,
        });
    }
    #[expect(
        clippy::needless_range_loop,
        reason = "each position indexes several per-vertex arrays and the geometry at once"
    )]
    for position in 0..vertex_count {
        if mismatched_vertex[position] || !geometry.vtags[position].boundary {
            continue;
        }
        if linear.linear_boundaries() {
            mismatched_vertex[position] = true;
            for &edge in &geometry.vertex_edges[position] {
                if geometry.etags[edge].boundary {
                    linear_edges[edge] = true;
                }
            }
        } else if geometry.vertex_faces[position].len() == 1 && make_smooth_corners_sharp {
            mismatched_vertex[position] = true;
        }
    }

    let mut map = ValueMap {
        corners: input
            .corners
            .iter()
            .map(|corners| vec![0; corners.len()])
            .collect(),
        vertex: Vec::new(),
        uvs: Vec::new(),
    };
    let mut sharpness = Vec::new();
    let mut non_manifold_values = Vec::new();
    let mut fvar_tags = Vec::new();
    let mut dependent_sources = Vec::new();
    for (position, vertex) in vertices.iter().enumerate() {
        let Some(vertex) = vertex else { continue };
        let tag = &geometry.vtags[position];
        let count = vertex.siblings.len();
        let boundary = vertex.fan.boundary;
        let processed = mismatched_vertex[position] || (vertex.non_manifold && count > 1);
        let first = map.uvs.len();
        if processed {
            let spans = value_spans(geometry, vertex, &mismatch, count);
            let mut all_sharp = linear.linear_boundaries()
                || is_infinite(geometry.vertex_sharpness[position])
                || vertex.non_manifold
                || (linear.dependent() && count > 2)
                || (linear.sharpen_darts() && count == 1 && !boundary);
            let mut has_dependent = false;
            if !all_sharp && linear.dependent() && count == 2 {
                let sharpen_all = spans[0].inf > 0
                    || spans[1].inf > 0
                    || spans[0].disct > 0
                    || spans[1].disct > 0
                    || (linear.sharpen_both_if_one_corner()
                        && (spans[0].size == 1 || spans[1].size == 1));
                all_sharp |= sharpen_all;
                has_dependent = (spans[0].semi > 0) != (spans[1].semi > 0);
            }
            for (s, span) in spans.iter().enumerate() {
                let disjoint = span.disct > 0;
                let sharp_value = disjoint
                    || all_sharp
                    || span.inf > 0
                    || (span.size == 1 && linear.corners_sharp());
                let dependent = has_dependent && span.semi == 0 && !sharp_value;
                let source_sibling = usize::from(spans[0].semi == 0);
                let dependency_source = dependent.then_some(first + source_sibling);
                let sharpness_value = if sharp_value {
                    SHARPNESS_INFINITE
                } else if tag.semi_sharp {
                    geometry.vertex_sharpness[position]
                } else if dependent {
                    span_semi_sharpness(geometry, vertex, position, source_sibling)
                        .unwrap_or(geometry.vertex_sharpness[position])
                } else {
                    0.0
                };
                let fvar_tag = if disjoint {
                    FVarTag {
                        mismatch: true,
                        non_manifold: true,
                        ..FVarTag::default()
                    }
                } else {
                    let is_inf_sharp =
                        all_sharp || span.inf > 0 || (span.size == 1 && linear.corners_sharp());
                    let xordinary = if span.size == 1 {
                        !is_inf_sharp
                    } else {
                        span.size != 2
                    };
                    let mut fvar_tag = FVarTag {
                        mismatch: true,
                        xordinary,
                        inf_sharp_edges: span.inf > 0,
                        inf_irregular: if span.inf > 0 {
                            span.size.saturating_sub(span.inf) > 1
                        } else if is_inf_sharp {
                            span.size > 1
                        } else {
                            xordinary
                        },
                        ..FVarTag::default()
                    };
                    if !is_inf_sharp {
                        if span.semi > 0 || tag.semi_sharp || dependent {
                            fvar_tag.semi_sharp = true;
                        } else {
                            fvar_tag.crease = true;
                        }
                    }
                    fvar_tag
                };
                map.vertex.push(position);
                map.uvs.push(input.uvs[vertex.siblings[s]]);
                sharpness.push(sharpness_value);
                non_manifold_values.push(disjoint);
                fvar_tags.push(fvar_tag);
                dependent_sources.push(dependency_source);
            }
        } else {
            map.vertex.push(position);
            map.uvs.push(input.uvs[vertex.siblings[0]]);
            sharpness.push(geometry.vertex_sharpness[position]);
            non_manifold_values.push(tag.non_manifold);
            fvar_tags.push(FVarTag::default());
            dependent_sources.push(None);
        }
        for (i, &face) in vertex.fan.faces.iter().enumerate() {
            let corner = geometry.corner_of(face, position);
            map.corners[face][corner] = first + vertex.sibling[i];
        }
    }
    (
        map,
        sharpness,
        non_manifold_values,
        fvar_tags,
        dependent_sources,
        linear_edges,
    )
}

/// Position topology marks edges traversed in the same direction by both incident faces as
/// non-manifold. The generic Level tags only count edges with more than two faces.
fn is_inconsistently_wound_edge(level: &Level, edge: usize) -> bool {
    let [first, _] = level.edges[edge];
    let direction = |face: usize| {
        level.face_edges[face]
            .iter()
            .position(|&candidate| candidate == edge)
            .map(|corner| level.faces[face][corner] == first)
    };
    if let &[face_a, face_b] = level.edge_faces[edge].as_slice() {
        matches!(
            (direction(face_a), direction(face_b)),
            (Some(a), Some(b)) if a == b
        )
    } else {
        false
    }
}

/// Marks both endpoints of `edge` as having a discontinuous face-varying topology.
fn mark_endpoints(geometry: &Level, edge: usize, marked: &mut [bool]) {
    let [a, b] = geometry.edges[edge];
    marked[a] = true;
    marked[b] = true;
}
