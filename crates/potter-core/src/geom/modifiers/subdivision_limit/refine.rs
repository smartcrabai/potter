// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from OpenSubdiv (https://graphics.pixar.com/opensubdiv) vtr/refinement.cpp, vtr/quadRefinement.cpp, far/topologyRefiner.cpp, sdc/scheme.h, far/primvarRefiner.h, Apache-2.0.

//! Sparse Catmull-Clark refinement: feature selection, child topology, child tags, child sharpness,
//! and child positions. Only the selected faces are split, together with the neighbourhood their
//! limit needs. Faces that are not selected keep their parent resolution as leaves.

use std::ops::{Add, Mul};

use super::level::{
    ETag, Level, RULE_CORNER, RULE_CREASE, RULE_DART, RULE_SMOOTH, VTag, decrement, is_infinite,
    is_semi_sharp, is_sharp, vertex_rule,
};

/// Positions and face-varying values share the refinement; both are affine combinations.
pub(super) trait Point: Copy + Default + Add<Output = Self> + Mul<f64, Output = Self> {}

impl<T: Copy + Default + Add<Output = T> + Mul<f64, Output = T>> Point for T {}

/// One refined level: the child topology, its positions, and the child quads of every parent face.
pub(super) struct Refined<T> {
    pub(super) level: Level,
    pub(super) positions: Vec<T>,
    /// `face_children[f][j]` is the child quad at corner `j` of parent face `f`, when it exists.
    pub(super) face_children: Vec<Vec<Option<usize>>>,
    /// For a dependent-sharp fvar vertex, the sibling value supplying its fractional crease weight.
    pub(super) dependent_source: Vec<Option<usize>>,
}

/// Feature-based selection of parent faces (OpenSubdiv `FeatureMask` with Blender's options).
pub(super) fn select_features(level: &Level, depth: usize) -> Vec<bool> {
    (0..level.faces.len())
        .map(|face| does_face_have_features(level, face, depth))
        .collect()
}

fn does_face_have_features(level: &Level, face: usize, depth: usize) -> bool {
    let corners = &level.faces[face];
    if corners.len() != 4 {
        return true;
    }
    let tags: Vec<VTag> = corners.iter().map(|&v| level.vtags[v]).collect();
    let combined = VTag::combine(&tags);
    if combined.incid_irreg_face {
        return true;
    }
    if combined.incomplete {
        return false;
    }
    if combined.non_manifold {
        return true;
    }
    if combined.xordinary {
        if combined.rule == RULE_SMOOTH {
            return true;
        }
        if depth < 2 && tags.iter().any(|t| t.xordinary && t.rule == RULE_SMOOTH) {
            return true;
        }
    }
    if combined.rule == RULE_SMOOTH {
        return false;
    }
    if combined.semi_sharp || combined.semi_sharp_edges {
        return true;
    }
    if combined.inf_sharp || combined.inf_sharp_edges {
        if combined.rule & RULE_SMOOTH != 0 {
            return inf_sharp_tag_has_features(&combined);
        }
        return tags
            .iter()
            .filter(|t| t.rule & RULE_SMOOTH == 0)
            .any(inf_sharp_tag_has_features);
    }
    false
}

/// An inf-sharp irregular corner, crease, or dart is a feature. Regular inf-sharp creases and
/// corners are not selected because `useInfSharpPatch` keeps them inside regular patches.
fn inf_sharp_tag_has_features(tag: &VTag) -> bool {
    tag.inf_irregular && tag.rule & (RULE_CORNER | RULE_CREASE | RULE_DART) != 0
}

/// A child position is the parent-level mask applied to the parent positions and the child face
/// centers. Each mask is stored by the parent's incidence order: the vertex, its incident edges'
/// far ends, and its incident faces' centers.
#[derive(Clone)]
struct VertexMask {
    vertex: f64,
    edges: Vec<f64>,
    faces: Vec<f64>,
}

impl VertexMask {
    fn identity(edges: usize, faces: usize) -> VertexMask {
        VertexMask {
            vertex: 1.0,
            edges: vec![0.0; edges],
            faces: vec![0.0; faces],
        }
    }

    /// `self * weight + other * other_weight`.
    fn blend(&self, weight: f64, other: &VertexMask, other_weight: f64) -> VertexMask {
        VertexMask {
            vertex: self.vertex * weight + other.vertex * other_weight,
            edges: self
                .edges
                .iter()
                .zip(&other.edges)
                .map(|(a, b)| a * weight + b * other_weight)
                .collect(),
            faces: self
                .faces
                .iter()
                .zip(&other.faces)
                .map(|(a, b)| a * weight + b * other_weight)
                .collect(),
        }
    }
}

/// Catmull-Clark smooth vertex mask: `(n-2)/n` on the vertex, `1/n^2` on each edge and face.
fn smooth_vertex_mask(edges: usize, faces: usize) -> VertexMask {
    let n = faces as f64;
    let weight = 1.0 / (n * n);
    VertexMask {
        vertex: (n - 2.0) / n,
        edges: vec![weight; edges],
        faces: vec![weight; faces],
    }
}

/// Crease vertex mask over the two sharp incident edges: `3/4` on the vertex, `1/8` on each end.
fn crease_vertex_mask(edges: usize, faces: usize, sharp: &[bool]) -> VertexMask {
    let mut mask = VertexMask {
        vertex: 0.75,
        edges: vec![0.0; edges],
        faces: vec![0.0; faces],
    };
    let mut ends = sharp
        .iter()
        .enumerate()
        .filter(|(_, is_sharp_edge)| **is_sharp_edge)
        .map(|(index, _)| index);
    if let (Some(first), Some(last)) = (ends.next(), ends.last()) {
        mask.edges[first] = 0.125;
        mask.edges[last] = 0.125;
    }
    mask
}

/// Mask for a vertex under one rule. `sharp` marks the incident edges that are sharp under the rule's
/// sharpness (the parent's for the parent rule, the child's for the child rule).
fn rule_mask(rule: u8, edges: usize, faces: usize, sharp: &[bool]) -> VertexMask {
    match rule {
        RULE_SMOOTH | RULE_DART => smooth_vertex_mask(edges, faces),
        RULE_CREASE => crease_vertex_mask(edges, faces, sharp),
        _ => VertexMask::identity(edges, faces),
    }
}

/// Fractional weight of the parent rule at a vertex whose rule changes between levels (uniform
/// creasing; OpenSubdiv `Crease::ComputeFractionalWeightAtVertex`).
fn fractional_weight(parent_vertex: f64, child_vertex: f64, parent_edges: &[f64]) -> f64 {
    let mut count = 0.0;
    let mut sum = 0.0;
    if is_sharp(parent_vertex) && !is_sharp(child_vertex) {
        count += 1.0;
        sum += parent_vertex;
    }
    for &sharpness in parent_edges {
        if is_sharp(sharpness) && sharpness <= 1.0 {
            count += 1.0;
            sum += sharpness;
        }
    }
    if count == 0.0 {
        0.0
    } else {
        (sum / count).min(1.0)
    }
}

/// Child sharpness of an edge half: infinite stays infinite, semi-sharp decays by one level.
fn child_edge_sharpness(parent: f64) -> f64 {
    if is_infinite(parent) {
        parent
    } else if is_semi_sharp(parent) {
        decrement(parent)
    } else {
        0.0
    }
}

/// Child sharpness of a vertex child (its vertex-level sharpness decays like an edge).
fn child_vertex_sharpness(parent_vertex: &VTag, parent_sharpness: f64) -> f64 {
    if parent_vertex.inf_sharp {
        super::level::SHARPNESS_INFINITE
    } else if is_semi_sharp(parent_sharpness) {
        decrement(parent_sharpness)
    } else {
        0.0
    }
}

/// Which parent entity a child vertex refines.
#[derive(Clone, Copy)]
enum Source {
    Vertex(usize),
    Edge(usize),
    Face(usize),
}

/// Child index tables derived from the selection.
struct Keys {
    vertex: Vec<Option<usize>>,
    edge: Vec<Option<usize>>,
    face: Vec<Option<usize>>,
    /// `half[e][s]`: child edge from parent endpoint `s` of `e` to the child edge vertex.
    half: Vec<[Option<usize>; 2]>,
    /// `spoke[f][k]`: child edge from the face center to the child vertex of parent edge `k`.
    spoke: Vec<Vec<Option<usize>>>,
    selected_edge: Vec<bool>,
}

fn child_keys(parent: &Level, selected: &[bool]) -> Keys {
    let mut selected_vertex = vec![false; parent.vertex_count()];
    let mut selected_edge = vec![false; parent.edge_count()];
    for (f, face) in parent.faces.iter().enumerate() {
        if selected[f] {
            for &v in face {
                selected_vertex[v] = true;
            }
            for &e in &parent.face_edges[f] {
                selected_edge[e] = true;
            }
        }
    }
    let mut next_vertex = 0usize;
    let mut take = |exists: bool| -> Option<usize> {
        exists.then(|| {
            next_vertex += 1;
            next_vertex - 1
        })
    };
    let mut next_edge = 0usize;
    let mut take_edge = |exists: bool| -> Option<usize> {
        exists.then(|| {
            next_edge += 1;
            next_edge - 1
        })
    };
    let vertex = (0..parent.vertex_count())
        .map(|v| take(selected_vertex[v]))
        .collect::<Vec<_>>();
    let edge = (0..parent.edge_count())
        .map(|e| {
            let [a, b] = parent.edges[e];
            take(selected_edge[e] || selected_vertex[a] || selected_vertex[b])
        })
        .collect::<Vec<_>>();
    let face = (0..parent.faces.len())
        .map(|f| {
            let any_corner = parent.faces[f].iter().any(|&v| selected_vertex[v]);
            take(selected[f] || any_corner)
        })
        .collect::<Vec<_>>();
    let half = (0..parent.edge_count())
        .map(|e| {
            let [a, b] = parent.edges[e];
            [
                take_edge(selected_edge[e] || selected_vertex[a]),
                take_edge(selected_edge[e] || selected_vertex[b]),
            ]
        })
        .collect::<Vec<_>>();
    let spoke = (0..parent.faces.len())
        .map(|f| {
            let corners = &parent.faces[f];
            let n = corners.len();
            (0..n)
                .map(|k| {
                    let exists = selected[f]
                        || selected_vertex[corners[k]]
                        || selected_vertex[corners[(k + 1) % n]];
                    take_edge(exists)
                })
                .collect()
        })
        .collect::<Vec<Vec<Option<usize>>>>();
    Keys {
        vertex,
        edge,
        face,
        half,
        spoke,
        selected_edge,
    }
}

/// Average of a parent face's corners: the child face center.
fn face_center<T: Point>(parent: &Level, positions: &[T], face: usize) -> T {
    let corners = &parent.faces[face];
    let weight = 1.0 / corners.len() as f64;
    corners
        .iter()
        .map(|&v| positions[v] * weight)
        .reduce(|acc, point| acc + point)
        .unwrap_or(positions[0] * 0.0)
}

/// Child edge-midpoint mask over the parent edge's endpoints and faces (smooth, crease or blended).
fn edge_child_position<T: Point>(
    parent: &Level,
    positions: &[T],
    edge: usize,
    child_rule: u8,
    face_points: &[T],
) -> T {
    let sharpness = parent.edge_sharpness[edge];
    let [a, b] = parent.edges[edge];
    let faces = &parent.edge_faces[edge];
    let n = faces.len() as f64;
    let (vertex_weight, face_weight, face_scale) = if sharpness <= 0.0 {
        (0.25, 0.5 / n, 1.0)
    } else if child_rule == RULE_CREASE {
        (0.5, 0.0, 0.0)
    } else {
        let parent_weight = sharpness.min(1.0);
        let child_weight = 1.0 - parent_weight;
        (
            parent_weight * 0.5 + child_weight * 0.25,
            0.5 / n,
            child_weight,
        )
    };
    let mut point = positions[a] * vertex_weight + positions[b] * vertex_weight;
    for &face in faces {
        point = point + face_points[face] * (face_weight * face_scale);
    }
    point
}

/// Child vertex position: the parent rule's mask, blended to the child rule when they differ.
fn vertex_child_position<T: Point>(
    parent: &Level,
    positions: &[T],
    vertex: usize,
    child_rule: u8,
    child_sharpness: f64,
    child_edge_sharpness: &[f64],
    dependent_source: Option<usize>,
    face_points: &[T],
) -> T {
    let parent_rule = parent.vtags[vertex].rule;
    let parent_edges = &parent.vertex_edges[vertex];
    let parent_faces = &parent.vertex_faces[vertex];
    let parent_edge_sharp: Vec<f64> = parent_edges
        .iter()
        .map(|&e| parent.edge_sharpness[e])
        .collect();
    let parent_sharp: Vec<bool> = parent_edge_sharp.iter().map(|&s| is_sharp(s)).collect();
    let parent_vertex_sharpness = parent.vertex_sharpness[vertex];
    let mut mask = rule_mask(
        parent_rule,
        parent_edges.len(),
        parent_faces.len(),
        &parent_sharp,
    );
    if parent_rule != RULE_SMOOTH && parent_rule != RULE_DART && child_rule != parent_rule {
        let child_sharp: Vec<bool> = child_edge_sharpness.iter().map(|&s| is_sharp(s)).collect();
        let child_mask = rule_mask(
            child_rule,
            parent_edges.len(),
            parent_faces.len(),
            &child_sharp,
        );
        let weight = if let Some(source) = dependent_source {
            let source_edges: Vec<f64> = parent.vertex_edges[source]
                .iter()
                .map(|&edge| parent.edge_sharpness[edge])
                .collect();
            let source_child_sharpness =
                child_vertex_sharpness(&parent.vtags[source], parent.vertex_sharpness[source]);
            fractional_weight(
                parent.vertex_sharpness[source],
                source_child_sharpness,
                &source_edges,
            )
        } else {
            fractional_weight(
                parent_vertex_sharpness,
                child_sharpness,
                &parent_edge_sharp,
            )
        };
        mask = mask.blend(weight, &child_mask, 1.0 - weight);
    }
    let mut point = positions[vertex] * mask.vertex;
    for (index, &edge) in parent_edges.iter().enumerate() {
        let far = parent.far_vertex(edge, vertex);
        point = point + positions[far] * mask.edges[index];
    }
    for (index, &face) in parent_faces.iter().enumerate() {
        point = point + face_points[face] * mask.faces[index];
    }
    point
}

/// Refines `parent` by the faces selected in `selected`. Child positions use the masks of the
/// parent's rules; the child keeps the parent's resolution wherever no selected face needs it.
pub(super) fn refine<T: Point>(
    parent: &Level,
    positions: &[T],
    selected: &[bool],
    depth: usize,
) -> Refined<T> {
    let keys = child_keys(parent, selected);
    let quads: Vec<Vec<bool>> = (0..parent.faces.len())
        .map(|f| {
            parent.faces[f]
                .iter()
                .map(|&v| selected[f] || keys.vertex[v].is_some())
                .collect()
        })
        .collect();
    assemble(parent, positions, selected, depth, &keys, &quads, false, &[])
}

/// Refines a face-varying channel over the child quads of a position refinement: `quads[f][j]` tells
/// whether face `f` of `parent` has a child quad at corner `j`. A child value, edge, or face exists
/// wherever a child quad uses it, so the channel has exactly the position refinement's quads.
pub(super) fn refine_values<T: Point>(
    parent: &Level,
    positions: &[T],
    selected: &[bool],
    depth: usize,
    quads: &[Vec<bool>],
    linear: bool,
    dependent_sources: &[Option<usize>],
) -> Refined<T> {
    let keys = value_keys(parent, quads);
    assemble(
        parent,
        positions,
        selected,
        depth,
        &keys,
        quads,
        linear,
        dependent_sources,
    )
}
/// Refines a face-varying channel around its own features and the position surface's selected faces.
/// This reproduces the combined feature mask OpenSubdiv uses when FVar channels are present.
/// Position refinement remains unchanged; additional FVar features may add their own child quads.
pub(super) fn refine_values_adaptive<T: Point>(
    parent: &Level,
    positions: &[T],
    depth: usize,
    linear: bool,
    dependent_sources: &[Option<usize>],
    position_selected: &[bool],
) -> Option<(Vec<bool>, Refined<T>)> {
    let mut selected = select_features(parent, depth);
    if selected.len() == position_selected.len() {
        for (selected, &position_selected) in selected.iter_mut().zip(position_selected) {
            *selected |= position_selected;
        }
    }
    if !selected.iter().any(|&selected| selected) {
        return None;
    }
    let initial_quads: Vec<Vec<bool>> = parent
        .faces
        .iter()
        .enumerate()
        .map(|(face, corners)| vec![selected[face]; corners.len()])
        .collect();
    let initial_keys = value_keys(parent, &initial_quads);
    let quads: Vec<Vec<bool>> = parent
        .faces
        .iter()
        .enumerate()
        .map(|(face, corners)| {
            corners
                .iter()
                .map(|&vertex| selected[face] || initial_keys.vertex[vertex].is_some())
                .collect()
        })
        .collect();
    let keys = value_keys(parent, &quads);
    let child = assemble(
        parent,
        positions,
        &selected,
        depth,
        &keys,
        &quads,
        linear,
        dependent_sources,
    );
    Some((selected, child))
}


/// Child indices of a face-varying channel: the children that its child quads use, numbered in the
/// order `child_keys` uses.
fn value_keys(parent: &Level, quads: &[Vec<bool>]) -> Keys {
    let mut used_vertex = vec![false; parent.vertex_count()];
    let mut used_edge = vec![false; parent.edge_count()];
    let mut used_face = vec![false; parent.faces.len()];
    let mut used_half = vec![[false; 2]; parent.edge_count()];
    let mut used_spoke: Vec<Vec<bool>> = parent
        .faces
        .iter()
        .map(|corners| vec![false; corners.len()])
        .collect();
    for (f, corners) in parent.faces.iter().enumerate() {
        let n = corners.len();
        for j in 0..n {
            if !quads[f][j] {
                continue;
            }
            let jp = (j + n - 1) % n;
            let (ej, ep) = (parent.face_edges[f][j], parent.face_edges[f][jp]);
            used_vertex[corners[j]] = true;
            used_face[f] = true;
            used_edge[ej] = true;
            used_edge[ep] = true;
            used_half[ej][usize::from(parent.edges[ej][0] != corners[j])] = true;
            used_half[ep][usize::from(parent.edges[ep][0] != corners[j])] = true;
            used_spoke[f][j] = true;
            used_spoke[f][jp] = true;
        }
    }
    // Vertex, edge, and face children share one numbering; halves and spokes another.
    let mut vertex_children = 0usize;
    let mut take_vertex = |used: bool| -> Option<usize> {
        used.then(|| {
            vertex_children += 1;
            vertex_children - 1
        })
    };
    let vertex: Vec<Option<usize>> = used_vertex.iter().map(|&used| take_vertex(used)).collect();
    let edge: Vec<Option<usize>> = used_edge.iter().map(|&used| take_vertex(used)).collect();
    let face: Vec<Option<usize>> = used_face.iter().map(|&used| take_vertex(used)).collect();
    let mut edge_children = 0usize;
    let mut take_edge = |used: bool| -> Option<usize> {
        used.then(|| {
            edge_children += 1;
            edge_children - 1
        })
    };
    let half: Vec<[Option<usize>; 2]> = used_half
        .iter()
        .map(|sides| [take_edge(sides[0]), take_edge(sides[1])])
        .collect();
    let spoke: Vec<Vec<Option<usize>>> = used_spoke
        .iter()
        .map(|sides| sides.iter().map(|&used| take_edge(used)).collect())
        .collect();
    Keys {
        vertex,
        edge,
        face,
        half,
        spoke,
        selected_edge: used_edge,
    }
}

/// Builds the child level of `parent` from its child quads: the child edges are the halves of parent
/// edges, then the spokes, and the child positions and tags follow the parent's rules.
fn assemble<T: Point>(
    parent: &Level,
    positions: &[T],
    selected: &[bool],
    depth: usize,
    keys: &Keys,
    quads: &[Vec<bool>],
    linear: bool,
    dependent_sources: &[Option<usize>],
) -> Refined<T> {
    let face_points: Vec<T> = (0..parent.faces.len())
        .map(|f| face_center(parent, positions, f))
        .collect();

    // Child edges: halves of parent edges first, then the spokes from face centers.
    let mut child_edges: Vec<[usize; 2]> = Vec::new();
    let mut edge_sharpness_out: Vec<f64> = Vec::new();
    let mut child_etags: Vec<ETag> = Vec::new();
    for e in 0..parent.edge_count() {
        for side in 0..2 {
            if let Some(index) = keys.half[e][side] {
                debug_assert_eq!(index, child_edges.len());
                let vertex = parent.edges[e][side];
                let sharpness = child_edge_sharpness(parent.edge_sharpness[e]);
                child_edges.push([
                    keys.vertex[vertex].expect("half edge has a child vertex"),
                    keys.edge[e].expect("half edge has an edge child"),
                ]);
                edge_sharpness_out.push(sharpness);
                child_etags.push(ETag {
                    non_manifold: parent.etags[e].non_manifold,
                    boundary: parent.etags[e].boundary,
                    inf_sharp: is_infinite(sharpness),
                    semi_sharp: is_semi_sharp(sharpness),
                    linear: parent.etags[e].linear,
                });
            }
        }
    }
    let mut spoke_index = vec![Vec::new(); parent.faces.len()];
    for f in 0..parent.faces.len() {
        spoke_index[f] = vec![None; parent.faces[f].len()];
        for k in 0..parent.faces[f].len() {
            if let Some(index) = keys.spoke[f][k] {
                debug_assert_eq!(index, child_edges.len());
                spoke_index[f][k] = Some(index);
                child_edges.push([
                    keys.face[f].expect("spoke has a face center"),
                    keys.edge[parent.face_edges[f][k]].expect("spoke has an edge child"),
                ]);
                let sharpness = if linear {
                    super::level::SHARPNESS_INFINITE
                } else {
                    0.0
                };
                edge_sharpness_out.push(sharpness);
                child_etags.push(ETag {
                    inf_sharp: linear,
                    ..ETag::default()
                });
            }
        }
    }

    // Child faces: child quad j is [V_j, E_j, C_f, E_{j-1}] with edges
    // [half(e_j, V_j), spoke_j, spoke_{j-1}, half(e_{j-1}, V_j)].
    let mut child_faces: Vec<Vec<usize>> = Vec::new();
    let mut child_face_edges: Vec<Vec<usize>> = Vec::new();
    let mut face_children = Vec::with_capacity(parent.faces.len());
    for f in 0..parent.faces.len() {
        let corners = &parent.faces[f];
        let n = corners.len();
        let mut children = vec![None; n];
        for j in 0..n {
            if !quads[f][j] {
                continue;
            }
            let jp = (j + n - 1) % n;
            let e_j = parent.face_edges[f][j];
            let e_prev = parent.face_edges[f][jp];
            let side_j = usize::from(parent.edges[e_j][0] != corners[j]);
            let side_prev = usize::from(parent.edges[e_prev][0] != corners[j]);
            children[j] = Some(child_faces.len());
            child_faces.push(vec![
                keys.vertex[corners[j]].expect("child corner has a vertex child"),
                keys.edge[e_j].expect("child corner has an edge child"),
                keys.face[f].expect("child quad has a face center"),
                keys.edge[e_prev].expect("child corner has an edge child"),
            ]);
            child_face_edges.push(vec![
                keys.half[e_j][side_j].expect("child quad has its edge half"),
                spoke_index[f][j].expect("child quad has its spoke"),
                spoke_index[f][jp].expect("child quad has its spoke"),
                keys.half[e_prev][side_prev].expect("child quad has its edge half"),
            ]);
        }
        face_children.push(children);
    }

    // Child vertices: vertex children, then edge children, then face children.
    let vertex_total = keys
        .vertex
        .iter()
        .chain(&keys.edge)
        .chain(&keys.face)
        .flatten()
        .count();
    let mut sources = vec![None; vertex_total];
    for (v, index) in keys.vertex.iter().enumerate() {
        if let Some(index) = *index {
            sources[index] = Some(Source::Vertex(v));
        }
    }
    for (e, index) in keys.edge.iter().enumerate() {
        if let Some(index) = *index {
            sources[index] = Some(Source::Edge(e));
        }
    }
    for (f, index) in keys.face.iter().enumerate() {
        if let Some(index) = *index {
            sources[index] = Some(Source::Face(f));
        }
    }

    let mut vertex_tags = Vec::with_capacity(vertex_total);
    let mut vertex_sharpness = Vec::with_capacity(vertex_total);
    let mut child_positions: Vec<T> = Vec::with_capacity(vertex_total);
    let mut child_dependent_sources = vec![None; vertex_total];
    for (child_index, source) in sources
        .iter()
        .map(|s| s.expect("child vertex indices are dense"))
        .enumerate()
    {
        let (tag, sharpness, point) = match source {
            Source::Vertex(v) => {
                let dependent_source = dependent_sources.get(v).copied().flatten();
                child_dependent_sources[child_index] =
                    dependent_source.and_then(|source| keys.vertex[source]);
                let parent_tag = parent.vtags[v];
                let edges = &parent.vertex_edges[v];
                let halves: Vec<f64> = edges
                    .iter()
                    .map(|&e| child_edge_sharpness(parent.edge_sharpness[e]))
                    .collect();
                let inf_count = halves.iter().filter(|&&s| is_infinite(s)).count();
                let semi_count = halves.iter().filter(|&&s| is_semi_sharp(s)).count();
                let sharpness = child_vertex_sharpness(&parent_tag, parent.vertex_sharpness[v]);
                let rule = vertex_rule(sharpness, inf_count + semi_count);
                let tag = VTag {
                    semi_sharp: is_semi_sharp(sharpness),
                    semi_sharp_edges: semi_count > 0,
                    inf_sharp: is_infinite(sharpness),
                    inf_sharp_edges: inf_count > 0,
                    incid_irreg_face: false,
                    incomplete: parent_tag.incomplete,
                    rule,
                    ..parent_tag
                };
                let point = vertex_child_position(
                    parent,
                    positions,
                    v,
                    rule,
                    sharpness,
                    &halves,
                    dependent_source,
                    &face_points,
                );
                (tag, sharpness, point)
            }
            Source::Edge(e) => {
                let parent_sharp = parent.edge_sharpness[e];
                let child_sharp = child_edge_sharpness(parent_sharp);
                let inf = is_infinite(parent_sharp);
                let semi_edges = is_semi_sharp(parent_sharp) && is_sharp(child_sharp);
                // Outside FVarLinear::All, a linear FVar edge is a corner only at a non-manifold split.
                let corner = linear || (parent.etags[e].linear && parent.etags[e].non_manifold);
                let tag = VTag {
                    non_manifold: parent.etags[e].non_manifold,
                    boundary: parent.etags[e].boundary,
                    corner,
                    inf_sharp: corner,
                    semi_sharp_edges: semi_edges,
                    inf_sharp_edges: inf,
                    inf_sharp_crease: inf && !corner,
                    inf_irregular: inf && parent.etags[e].non_manifold,
                    incomplete: !keys.selected_edge[e],
                    rule: if corner {
                        RULE_CORNER
                    } else if inf || semi_edges {
                        RULE_CREASE
                    } else {
                        RULE_SMOOTH
                    },
                    ..VTag::default()
                };
                let point = edge_child_position(parent, positions, e, tag.rule, &face_points);
                (tag, if linear { super::level::SHARPNESS_INFINITE } else { 0.0 }, point)
            }
            Source::Face(f) => {
                let corners = parent.faces[f].len();
                let tag = VTag {
                    xordinary: depth == 0 && corners != 4,
                    incomplete: !selected[f],
                    rule: if linear { RULE_CORNER } else { RULE_SMOOTH },
                    inf_sharp: linear,
                    ..VTag::default()
                };
                (tag, if linear { super::level::SHARPNESS_INFINITE } else { 0.0 }, face_points[f])
            }
        };
        vertex_tags.push(tag);
        vertex_sharpness.push(sharpness);
        child_positions.push(point);
    }


    let mut level = Level::from_parts(child_faces, child_face_edges, child_edges, vertex_total);
    level.edge_sharpness = edge_sharpness_out;
    level.etags = child_etags;
    level.vertex_sharpness = vertex_sharpness;
    level.vtags = vertex_tags;
    Refined {
        level,
        positions: child_positions,
        face_children,
        dependent_source: child_dependent_sources,
    }
}
