// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from OpenSubdiv (https://graphics.pixar.com/opensubdiv) far/patchMap.cpp, far/patchTableFactory.cpp, far/topologyRefiner.cpp, Apache-2.0.
// Portions derived from Blender (https://www.blender.org) source/blender/blenkernel/intern/subdiv_eval.cc, GPL-2.0-or-later.

//! Blender's limit-surface evaluation: sparse adaptive refinement to the isolation depth, then a
//! patch per leaf quad (B-spline or Gregory), evaluated at ptex samples.

mod face_varying;
mod gregory;
mod level;
mod patch;
mod refine;

use std::collections::BTreeMap;

use crate::error::{ErrorCode, PotError, Result};
use glam::{DVec2, DVec3};

use self::{
    gregory::{control_points, corner_span, is_singular_edge, leaf_corner, leaf_corner_fvar},
    level::{Level, SHARPNESS_INFINITE, VTag},
    patch::{
        Row, boundary_mask, bspline_weights, bspline_weights_and_derivatives, gregory_weights,
        gregory_weights_and_derivatives, is_regular_leaf, regular_grid,
    },
    refine::{Point, Refined, refine, refine_values, refine_values_adaptive, select_features},
};
use super::subdivision_uv::{FvarLinear, ValueMap};

/// A grid sample: the base face, the ptex sub-face (corner index of an n-gon, 0 for a quad), and the
/// ptex coordinates within it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Sample {
    pub(super) face: usize,
    pub(super) ptex: usize,
    pub(super) u: f64,
    pub(super) v: f64,
}

/// Refined levels of one mesh, with the selection that produced each level from the previous one.
pub(super) struct LimitSurface {
    levels: Vec<Refined<DVec3>>,
    /// `selected[l][f]`: face `f` of level `l` was split into level `l + 1`.
    selected: Vec<Vec<bool>>,
    /// Maximum isolation depth shared by the position and face-varying patch builders.
    isolation: usize,
    /// Base faces that are n-gons, whose ptex sub-faces are the level-1 children.
    ngon_root: Vec<bool>,
    has_irregular_faces: bool,
}

#[derive(Clone, Copy)]
struct LocatedSample {
    level: usize,
    face: usize,
    s: f64,
    t: f64,
    /// Rows are leaf s/t; columns are ptex u/v.
    jacobian: [[f64; 2]; 2],
}

/// The base-mesh sharpness that a limit surface is built with.
pub(super) enum BaseSharpness<'a> {
    /// Blender's edge creases (`10 c^2`), with the non-manifold edges already infinitely sharp.
    /// `preserve_corners` sharpens single-face boundary corners; `non_manifold[v]` marks the
    /// vertices that are infinitely sharp corners because of non-manifold topology.
    Creases {
        edges: &'a BTreeMap<(usize, usize), f64>,
        non_manifold: &'a [bool],
        preserve_corners: bool,
    },
    /// SIMPLE subdivision. Blender refines it as Catmull-Clark with every edge and every vertex
    /// infinitely sharp (`BUGGY_SIMPLE_SCHEME_WORKAROUND` in `subdiv_converter_mesh.cc`).
    Simple,
}

/// Builds the adaptive refinement to the isolation depth (Blender's quality). Without
/// `use_limit_surface` Blender outputs the refined control positions, so this is not used there.
impl LimitSurface {
    /// Builds the refined levels for the base mesh with its sharpness and boundary/corner options.
    pub(super) fn build(
        faces: &[Vec<usize>],
        points: &[DVec3],
        sharpness: BaseSharpness,
        isolation: usize,
    ) -> LimitSurface {
        let mut base = Level::from_faces(faces.to_vec(), points.len());
        let (preserve_corners, non_manifold) = match sharpness {
            BaseSharpness::Creases {
                edges,
                non_manifold,
                preserve_corners,
            } => {
                for (edge, [a, b]) in base.edges.clone().into_iter().enumerate() {
                    base.edge_sharpness[edge] = edges.get(&(a, b)).copied().unwrap_or(0.0);
                }
                (preserve_corners, non_manifold.to_vec())
            }
            BaseSharpness::Simple => {
                base.edge_sharpness.fill(SHARPNESS_INFINITE);
                base.vertex_sharpness.fill(SHARPNESS_INFINITE);
                (false, vec![false; points.len()])
            }
        };
        base.derive_base_tags(preserve_corners, &non_manifold);
        let mut levels = vec![Refined {
            level: base,
            positions: points.to_vec(),
            face_children: Vec::new(),
            dependent_source: Vec::new(),
        }];
        let mut selected = Vec::new();
        for l in 0..isolation {
            let selection = select_features(&levels[l].level, l);
            if !selection.iter().any(|&s| s) {
                break;
            }
            let next = refine(&levels[l].level, &levels[l].positions, &selection, l);
            selected.push(selection);
            levels.push(next);
        }
        let ngon_root: Vec<bool> = faces.iter().map(|face| face.len() != 4).collect();
        let has_irregular_faces = ngon_root.iter().any(|&irregular| irregular);
        LimitSurface {
            levels,
            selected,
            isolation,
            ngon_root,
            has_irregular_faces,
        }
    }

    /// Evaluates the limit at a sample (Blender's `evaluateLimit`): the leaf that holds the sample,
    /// evaluated at its local coordinates.
    pub(super) fn evaluate(&self, sample: Sample) -> DVec3 {
        let located = self.locate(sample, false);
        let refined = &self.levels[located.level];
        leaf_point(
            &refined.level,
            &refined.positions,
            located.face,
            located.level,
            self.has_irregular_faces,
            located.s,
            located.t,
        )
    }

    /// Evaluates the limit point and its object-space derivatives with respect to the sample's
    /// ptex-local `u` and `v`, using OpenSubdiv's patch basis derivative rules.
    pub(super) fn point_and_derivatives(&self, sample: Sample) -> Result<(DVec3, DVec3, DVec3)> {
        let Some(&is_ngon) = self.ngon_root.get(sample.face) else {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "subdivision sample references a missing face",
            ));
        };
        let face = &self.levels[0].level.faces[sample.face];
        let ptex_count = if is_ngon { face.len() } else { 1 };
        if sample.ptex >= ptex_count
            || !sample.u.is_finite()
            || !sample.v.is_finite()
            || !(0.0..=1.0).contains(&sample.u)
            || !(0.0..=1.0).contains(&sample.v)
        {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "subdivision sample has invalid ptex coordinates",
            ));
        }
        let located = self.locate(sample, true);
        let refined = &self.levels[located.level];
        let (point, ds, dt) = leaf_point_and_derivatives(
            &refined.level,
            &refined.positions,
            located.face,
            located.level,
            self.has_irregular_faces,
            located.s,
            located.t,
        );
        let jacobian = located.jacobian;
        Ok((
            point,
            ds * jacobian[0][0] + dt * jacobian[1][0],
            ds * jacobian[0][1] + dt * jacobian[1][1],
        ))
    }

    /// The UV layer's value at every output loop, for the loops' `samples`, on this limit surface. The
    /// layer's values refine over the position levels and are evaluated on the same leaf, at the same
    /// local coordinates, as the position of each loop.
    pub(super) fn face_varying_loops(
        &self,
        linear: FvarLinear,
        preserve_corners: bool,
        map: &ValueMap,
        samples: &Samples,
    ) -> Vec<DVec2> {
        let mut levels = vec![face_varying::base(
            &self.levels[0].level,
            linear,
            preserve_corners,
            map,
        )];
        let mut selected = Vec::new();
        if linear == FvarLinear::All {
            for depth in 0..self.selected.len() {
                let quads: Vec<Vec<bool>> = self.levels[depth + 1]
                    .face_children
                    .iter()
                    .map(|children| children.iter().map(Option::is_some).collect())
                    .collect();
                let parent = &levels[depth];
                let child = refine_values(
                    &parent.level,
                    &parent.positions,
                    &self.selected[depth],
                    depth,
                    &quads,
                    true,
                    &parent.dependent_source,
                );
                selected.push(self.selected[depth].clone());
                levels.push(child);
            }
        } else {
            for depth in 0..self.isolation {
                let parent = levels.last().expect("base fvar level exists");
                let position_selected = self
                    .selected
                    .get(depth)
                    .map_or(&[][..], |selection| selection.as_slice());
                let Some((selection, child)) = refine_values_adaptive(
                    &parent.level,
                    &parent.positions,
                    depth,
                    false,
                    &parent.dependent_source,
                    position_selected,
                ) else {
                    break;
                };
                selected.push(selection);
                levels.push(child);
            }
        }
        samples
            .corners
            .iter()
            .flatten()
            .map(|&sample| {
                let (level, face, s, t) = self.locate_fvar(sample, &levels, &selected);
                let values = &levels[level];
                if linear == FvarLinear::All {
                    bilinear(&values.level, &values.positions, face, s, t)
                } else {
                    leaf_fvar_point(
                        &values.level,
                        &values.positions,
                        face,
                        level,
                        self.has_irregular_faces,
                        s,
                        t,
                    )
                }
            })
            .collect()
    }

    /// The UV layer's value at every output loop when `use_limit_surface` is off: the layer is refined
    /// uniformly `levels` times, as the output positions are.
    pub(super) fn uniform_face_varying(
        &self,
        linear: FvarLinear,
        preserve_corners: bool,
        map: &ValueMap,
        levels: usize,
    ) -> Vec<DVec2> {
        let mut current = face_varying::base(&self.levels[0].level, linear, preserve_corners, map);
        for depth in 0..levels {
            let quads: Vec<Vec<bool>> = current
                .level
                .faces
                .iter()
                .map(|corners| vec![true; corners.len()])
                .collect();
            let selected = vec![true; quads.len()];
            current = refine_values(
                &current.level,
                &current.positions,
                &selected,
                depth,
                &quads,
                linear == FvarLinear::All,
                &current.dependent_source,
            );
        }
        current
            .level
            .faces
            .iter()
            .flatten()
            .map(|&value| current.positions[value])
            .collect()
    }

    /// The leaf that holds `sample` (Blender's quadtree lookup): its level and face, and the sample's
    /// local coordinates and ptex-to-leaf Jacobian.
    fn locate(&self, sample: Sample, track_jacobian: bool) -> LocatedSample {
        let (mut level, mut face, mut s, mut t) = if self.ngon_root[sample.face] {
            let child = self.levels[1].face_children[sample.face][sample.ptex]
                .expect("n-gon sub-faces are refined at the first level");
            (1, child, sample.u, sample.v)
        } else {
            (0, sample.face, sample.u, sample.v)
        };
        let mut jacobian = [[1.0, 0.0], [0.0, 1.0]];
        while level + 1 < self.levels.len() && self.selected[level][face] {
            let quadrant = match (s >= 0.5, t >= 0.5) {
                (false, false) => 0,
                (true, false) => 1,
                (true, true) => 2,
                (false, true) => 3,
            };
            let child = self.levels[level + 1].face_children[face][quadrant]
                .expect("a selected face has all of its child quads");
            let (child_s, child_t, child_jacobian) = child_coordinates(quadrant, s, t);
            if track_jacobian {
                jacobian = compose_jacobian(child_jacobian, jacobian);
            }
            (s, t) = (child_s, child_t);
            face = child;
            level += 1;
        }
        LocatedSample {
            level,
            face,
            s,
            t,
            jacobian,
        }
    }

    /// The leaf holding a sample in a channel's fvar-specific isolation tree.
    fn locate_fvar(
        &self,
        sample: Sample,
        levels: &[Refined<DVec2>],
        selected: &[Vec<bool>],
    ) -> (usize, usize, f64, f64) {
        let (mut level, mut face, mut s, mut t) = if self.ngon_root[sample.face] {
            let child = levels[1].face_children[sample.face][sample.ptex]
                .expect("n-gon fvar sub-faces are refined at the first level");
            (1, child, sample.u, sample.v)
        } else {
            (0, sample.face, sample.u, sample.v)
        };
        while level + 1 < levels.len() && selected[level][face] {
            let quadrant = match (s >= 0.5, t >= 0.5) {
                (false, false) => 0,
                (true, false) => 1,
                (true, true) => 2,
                (false, true) => 3,
            };
            let child = levels[level + 1].face_children[face][quadrant]
                .expect("a selected fvar face has all child quads");
            (s, t, _) = child_coordinates(quadrant, s, t);
            face = child;
            level += 1;
        }
        (level, face, s, t)
    }
}

/// Local coordinates and their Jacobian in child quad `quadrant` of its parent, in the child's own
/// frame (whose corner 0 is the parent's corner `quadrant`).
fn child_coordinates(quadrant: usize, s: f64, t: f64) -> (f64, f64, [[f64; 2]; 2]) {
    match quadrant {
        0 => (2.0 * s, 2.0 * t, [[2.0, 0.0], [0.0, 2.0]]),
        1 => (2.0 * t, 2.0 - 2.0 * s, [[0.0, 2.0], [-2.0, 0.0]]),
        2 => (2.0 - 2.0 * s, 2.0 - 2.0 * t, [[-2.0, 0.0], [0.0, -2.0]]),
        _ => (2.0 - 2.0 * t, 2.0 * s, [[0.0, -2.0], [2.0, 0.0]]),
    }
}

/// Compose a child-local mapping with its parent's ptex-local mapping.
fn compose_jacobian(child: [[f64; 2]; 2], parent: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            child[0][0] * parent[0][0] + child[0][1] * parent[1][0],
            child[0][0] * parent[0][1] + child[0][1] * parent[1][1],
        ],
        [
            child[1][0] * parent[0][0] + child[1][1] * parent[1][0],
            child[1][0] * parent[0][1] + child[1][1] * parent[1][1],
        ],
    ]
}

/// Evaluates a leaf quad at local coordinates: a B-spline when the patch is regular, otherwise the
/// Gregory endcap.
fn leaf_point<T: Point>(
    level: &Level,
    positions: &[T],
    face: usize,
    depth: usize,
    has_irregular_faces: bool,
    s: f64,
    t: f64,
) -> T {
    if is_regular_leaf(level, face, depth, has_irregular_faces) {
        let mask = boundary_mask(level, face);
        let grid = regular_grid(level, face, mask);
        let weights = bspline_weights(s, t, mask);
        weighted_rows(&grid, &weights, positions)
    } else {
        let corners = std::array::from_fn(|c| leaf_corner(level, face, level.faces[face][c]));
        let rows = control_points(&corners);
        let weights = gregory_weights(s, t);
        weighted_rows(&rows, &weights, positions)
    }
}

/// Evaluates a position patch and OpenSubdiv's default (approximate Gregory) derivatives in local
/// patch coordinates. The point-only path remains separate and does not compute unused derivatives.
fn leaf_point_and_derivatives(
    level: &Level,
    positions: &[DVec3],
    face: usize,
    depth: usize,
    has_irregular_faces: bool,
    s: f64,
    t: f64,
) -> (DVec3, DVec3, DVec3) {
    if is_regular_leaf(level, face, depth, has_irregular_faces) {
        let mask = boundary_mask(level, face);
        let grid = regular_grid(level, face, mask);
        let (weights, ds, dt) = bspline_weights_and_derivatives(s, t, mask);
        (
            weighted_rows(&grid, &weights, positions),
            weighted_rows(&grid, &ds, positions),
            weighted_rows(&grid, &dt, positions),
        )
    } else {
        let corners = std::array::from_fn(|c| leaf_corner(level, face, level.faces[face][c]));
        let rows = control_points(&corners);
        let (weights, ds, dt) = gregory_weights_and_derivatives(s, t);
        (
            weighted_rows(&rows, &weights, positions),
            weighted_rows(&rows, &ds, positions),
            weighted_rows(&rows, &dt, positions),
        )
    }
}

/// Face-varying endcaps use OpenSubdiv's non-manifold span walk; position endcaps keep the
/// orientation-independent fan used by `leaf_point`.
fn leaf_fvar_point<T: Point>(
    level: &Level,
    positions: &[T],
    face: usize,
    depth: usize,
    has_irregular_faces: bool,
    s: f64,
    t: f64,
) -> T {
    if is_regular_fvar_leaf(level, face, depth, has_irregular_faces) {
        let mask = boundary_mask(level, face);
        let grid = regular_grid(level, face, mask);
        let weights = bspline_weights(s, t, mask);
        weighted_rows(&grid, &weights, positions)
    } else {
        let corners = std::array::from_fn(|c| leaf_corner_fvar(level, face, level.faces[face][c]));
        let rows = control_points(&corners);
        let weights = gregory_weights(s, t);
        weighted_rows(&rows, &weights, positions)
    }
}

/// OpenSubdiv's fvar `IsPatchRegular` adds a span test for non-manifold corners. The position
/// classifier remains unchanged; this channel-specific variant permits a regular endcap when the
/// non-manifold value's face span is regular.
fn is_regular_fvar_leaf(
    level: &Level,
    face: usize,
    depth: usize,
    has_irregular_faces: bool,
) -> bool {
    let corners = &level.faces[face];
    let tags: Vec<_> = corners.iter().map(|&vertex| level.vtags[vertex]).collect();
    let combined = VTag::combine(&tags);
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
    for (corner, tag) in tags.iter().enumerate() {
        if !(tag.non_manifold || tag.xordinary || tag.inf_irregular) {
            continue;
        }
        if tag.non_manifold {
            let num_span_faces = corner_span(level, face, corners[corner]).faces.len();
            let expected = if tag.inf_sharp { 1 } else { 2 };
            if num_span_faces != expected {
                return false;
            }
            continue;
        }
        if tag.xordinary {
            if !tag.inf_sharp_edges {
                return false;
            }
            if tag.boundary && !tag.inf_sharp {
                let previous = (corner + 3) % 4;
                let out_boundary = is_singular_edge(level, level.face_edges[face][corner]);
                let in_boundary = is_singular_edge(level, level.face_edges[face][previous]);
                if out_boundary && in_boundary {
                    continue;
                }
            }
        }
        if tag.inf_irregular {
            if !tag.inf_sharp_edges || (tag.inf_sharp_crease && tag.boundary) {
                return false;
            }
            let span_faces = corner_span(level, face, corners[corner]).faces.len();
            let expected = if tag.inf_sharp_crease { 2 } else { 1 };
            if span_faces != expected {
                return false;
            }
        }
    }
    true
}

/// Grid samples of one level: the sample of each vertex that some face uses, and of each corner of
/// every face in the level's face order. A ptex coordinate is only meaningful in its own sub-face,
/// so midpoints and centers are averaged within one sub-face.
#[derive(Clone, Debug)]
pub(super) struct Samples {
    pub(super) vertices: Vec<Option<Sample>>,
    pub(super) corners: Vec<Vec<Sample>>,
}

/// Corner coordinates of a quad sub-face, in corner order.
const QUAD_CORNERS: [(f64, f64); 4] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];

impl Samples {
    /// Samples of the base mesh. A quad's corners are its ptex corners; an n-gon's corner `k` is the
    /// corner of its sub-face `k`, and its center is the `(1, 1)` corner of sub-face 0.
    pub(super) fn base(faces: &[Vec<usize>], vertex_count: usize) -> Samples {
        let mut vertices = vec![None; vertex_count];
        let mut corners = Vec::with_capacity(faces.len());
        for (face, corner_vertices) in faces.iter().enumerate() {
            let mut face_corners = Vec::with_capacity(corner_vertices.len());
            for (k, &vertex) in corner_vertices.iter().enumerate() {
                let sample = if corner_vertices.len() == 4 {
                    let (u, v) = QUAD_CORNERS[k];
                    Sample {
                        face,
                        ptex: 0,
                        u,
                        v,
                    }
                } else {
                    Sample {
                        face,
                        ptex: k,
                        u: 0.0,
                        v: 0.0,
                    }
                };
                vertices[vertex].get_or_insert(sample);
                face_corners.push(sample);
            }
            corners.push(face_corners);
        }
        Samples { vertices, corners }
    }

    /// Samples of the next level, in the caller's point numbering: each old vertex keeps its sample,
    /// each new edge point (`edge_points`, keyed by sorted endpoints) takes the sample of its edge,
    /// and each new face point, after the edge points, takes its face center. `faces` are the
    /// current level's corner lists.
    pub(super) fn next(
        &self,
        faces: &[Vec<usize>],
        edge_points: &BTreeMap<(usize, usize), usize>,
        base_level: bool,
    ) -> Samples {
        let mut edge_samples = BTreeMap::new();
        for (face, corner_vertices) in faces.iter().enumerate() {
            let n = corner_vertices.len();
            for j in 0..n {
                let key = edge_key(corner_vertices[j], corner_vertices[(j + 1) % n]);
                edge_samples.entry(key).or_insert_with(|| {
                    if base_level && n != 4 {
                        Sample {
                            face,
                            ptex: j,
                            u: 1.0,
                            v: 0.0,
                        }
                    } else {
                        midpoint(self.corners[face][j], self.corners[face][(j + 1) % n])
                    }
                });
            }
        }
        let face_base = self.vertices.len() + edge_points.len();
        let mut vertices = vec![None; face_base + faces.len()];
        vertices[..self.vertices.len()].copy_from_slice(&self.vertices);
        for (key, &index) in edge_points {
            vertices[index] = edge_samples.get(key).copied();
        }
        let mut children = Vec::with_capacity(faces.len());
        for (face, corner_vertices) in faces.iter().enumerate() {
            let n = corner_vertices.len();
            let corners = &self.corners[face];
            vertices[face_base + face] = Some(if base_level && n != 4 {
                Sample {
                    face,
                    ptex: 0,
                    u: 1.0,
                    v: 1.0,
                }
            } else {
                center(corners)
            });
            for j in 0..n {
                let child = if base_level && n != 4 {
                    QUAD_CORNERS
                        .iter()
                        .map(|&(u, v)| Sample {
                            face,
                            ptex: j,
                            u,
                            v,
                        })
                        .collect()
                } else {
                    let previous = (j + n - 1) % n;
                    vec![
                        corners[j],
                        midpoint(corners[j], corners[(j + 1) % n]),
                        center(corners),
                        midpoint(corners[previous], corners[j]),
                    ]
                };
                children.push(child);
            }
        }
        Samples {
            vertices,
            corners: children,
        }
    }
}

/// Midpoint of two samples of one sub-face.
fn midpoint(a: Sample, b: Sample) -> Sample {
    Sample {
        face: a.face,
        ptex: a.ptex,
        u: 0.5 * (a.u + b.u),
        v: 0.5 * (a.v + b.v),
    }
}

/// Center of four corner samples of one sub-face.
fn center(corners: &[Sample]) -> Sample {
    let first = corners[0];
    let (u, v) = corners
        .iter()
        .fold((0.0, 0.0), |(u, v), corner| (u + corner.u, v + corner.v));
    Sample {
        face: first.face,
        ptex: first.ptex,
        u: u / corners.len() as f64,
        v: v / corners.len() as f64,
    }
}

/// Sorted edge key.
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// `sum_k weights[k] * (sum_j row_k[j].weight * positions[row_k[j].vertex])`.
fn weighted_rows<T: Point, const N: usize>(
    rows: &[Row; N],
    weights: &[f64; N],
    positions: &[T],
) -> T {
    let mut point = T::default();
    for (row, &weight) in rows.iter().zip(weights) {
        for &(vertex, w) in row {
            point = point + positions[vertex] * (weight * w);
        }
    }
    point
}

/// Bilinear value of a linear channel's leaf quad at local coordinates (OpenSubdiv's `QUADS` patches).
fn bilinear(level: &Level, positions: &[DVec2], face: usize, s: f64, t: f64) -> DVec2 {
    let weights = [(1.0 - s) * (1.0 - t), s * (1.0 - t), s * t, (1.0 - s) * t];
    level.faces[face]
        .iter()
        .zip(weights)
        .fold(DVec2::ZERO, |sum, (&value, weight)| {
            sum + positions[value] * weight
        })
}

#[cfg(test)]
mod tests {
    use super::{child_coordinates, compose_jacobian};

    fn child_quadrant(s: f64, t: f64) -> usize {
        match (s >= 0.5, t >= 0.5) {
            (false, false) => 0,
            (true, false) => 1,
            (true, true) => 2,
            (false, true) => 3,
        }
    }

    fn parent_coordinates(quadrant: usize, s: f64, t: f64) -> (f64, f64) {
        match quadrant {
            0 => (0.5 * s, 0.5 * t),
            1 => (1.0 - 0.5 * t, 0.5 * s),
            2 => (1.0 - 0.5 * s, 1.0 - 0.5 * t),
            _ => (0.5 * t, 1.0 - 0.5 * s),
        }
    }

    fn map_path(path: &[usize], mut s: f64, mut t: f64) -> (f64, f64) {
        for &quadrant in path {
            (s, t, _) = child_coordinates(quadrant, s, t);
        }
        (s, t)
    }

    #[test]
    fn child_jacobians_compose_for_every_three_level_quadrant_path() {
        let leaf = (0.31, 0.67);
        let epsilon = 1.0e-7;
        for encoded_path in 0..64 {
            let path = [
                encoded_path & 3,
                (encoded_path >> 2) & 3,
                (encoded_path >> 4) & 3,
            ];
            let mut root = leaf;
            for &quadrant in path.iter().rev() {
                root = parent_coordinates(quadrant, root.0, root.1);
            }

            let mut local = root;
            let mut jacobian = [[1.0, 0.0], [0.0, 1.0]];
            for quadrant in path {
                assert_eq!(child_quadrant(local.0, local.1), quadrant);
                let (s, t, child_jacobian) = child_coordinates(quadrant, local.0, local.1);
                jacobian = compose_jacobian(child_jacobian, jacobian);
                local = (s, t);
            }
            assert!((local.0 - leaf.0).abs() < 1.0e-12);
            assert!((local.1 - leaf.1).abs() < 1.0e-12);

            let du = map_path(&path, root.0 + epsilon, root.1);
            let dv = map_path(&path, root.0, root.1 + epsilon);
            assert!(((du.0 - local.0) / epsilon - jacobian[0][0]).abs() < 1.0e-8);
            assert!(((du.1 - local.1) / epsilon - jacobian[1][0]).abs() < 1.0e-8);
            assert!(((dv.0 - local.0) / epsilon - jacobian[0][1]).abs() < 1.0e-8);
            assert!(((dv.1 - local.1) / epsilon - jacobian[1][1]).abs() < 1.0e-8);
        }
    }
}
