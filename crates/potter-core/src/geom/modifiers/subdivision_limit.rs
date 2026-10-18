// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from OpenSubdiv (https://graphics.pixar.com/opensubdiv) far/patchMap.cpp, far/patchTableFactory.cpp, far/topologyRefiner.cpp, Apache-2.0.
// Portions derived from Blender (https://www.blender.org) source/blender/blenkernel/intern/subdiv_eval.cc, GPL-2.0-or-later.

//! Blender's limit-surface evaluation: sparse adaptive refinement to the isolation depth, then a
//! patch per leaf quad (B-spline or Gregory), evaluated at ptex samples. Without `use_limit_surface`
//! Blender outputs the uniformly refined control points instead (`UniformPoints`).

mod face_varying;
mod gregory;
mod level;
mod patch;
mod refine;

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};

use self::{
    gregory::{leaf_corner, leaf_corner_fvar},
    level::{Level, SHARPNESS_INFINITE},
    patch::{CornerBuilder, Patch},
    refine::{
        Children, Point, Refined, refine, refine_values, refine_values_adaptive, select_features,
    },
};
use super::Polygon;
use super::subdivision_uv::{FvarLinear, ValueMap};

/// A grid sample: the base face, the ptex sub-face (corner index of an n-gon, 0 for a quad), and the
/// ptex coordinates within it. Samples are in the canonical corner order of their base face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Sample {
    pub(super) face: usize,
    pub(super) ptex: usize,
    pub(super) u: f64,
    pub(super) v: f64,
}

impl Sample {
    /// The sample in Blender's loop frame, where the face's first loop is at corner `start`: canonical
    /// corner `k` is ptex corner `(k - start) mod size`. `size` is the corner count of the base face.
    pub(super) fn in_loop_frame(self, start: usize, size: usize) -> Sample {
        if size == 4 {
            // One quarter turn maps corner `k` of the unit square to corner `k - 1`.
            let (mut u, mut v) = (self.u, self.v);
            for _ in 0..start % 4 {
                (u, v) = (v, 1.0 - u);
            }
            Sample { u, v, ..self }
        } else {
            Sample {
                ptex: (self.ptex + size - start % size) % size,
                ..self
            }
        }
    }
}

/// The refined levels of one mesh, with the selection that produced each level from the previous one.
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

/// The base-mesh sharpness that a limit surface is built with.
pub(super) enum BaseSharpness<'a> {
    /// Blender's creases, already converted to sharpness (`10 c^2`). `edges` maps sorted endpoints to
    /// edge sharpness and `vertices[v]` is the sharpness of vertex `v`. Vertices on a loose edge are
    /// infinitely sharp. `non_manifold[v]` marks the vertices that are infinitely sharp corners because
    /// of non-manifold topology. `preserve_corners` sharpens single-face boundary corners.
    Creases {
        edges: &'a BTreeMap<(usize, usize), f64>,
        vertices: &'a [f64],
        non_manifold: &'a [bool],
        loose_edges: &'a [[usize; 2]],
        preserve_corners: bool,
    },
    /// SIMPLE subdivision. Blender refines it as Catmull-Clark with every edge and every vertex
    /// infinitely sharp (`BUGGY_SIMPLE_SCHEME_WORKAROUND` in `subdiv_converter_mesh.cc`).
    Simple,
}

/// The base level of a mesh: its topology with the sharpness and tags of `sharpness`, and the points.
fn base_level(polygons: &[Polygon], points: &[DVec3], sharpness: &BaseSharpness) -> Refined<DVec3> {
    let faces = polygons.iter().map(|(face, _)| face.clone()).collect();
    let mut level = Level::from_faces(faces, points.len());
    let no_non_manifold = vec![false; points.len()];
    let (preserve_corners, non_manifold) = match sharpness {
        BaseSharpness::Creases {
            edges,
            vertices,
            non_manifold,
            loose_edges,
            preserve_corners,
        } => {
            for edge in 0..level.edge_count() {
                let [a, b] = level.edges[edge];
                level.edge_sharpness[edge] = edges.get(&(a, b)).copied().unwrap_or(0.0);
            }
            level.vertex_sharpness.clone_from_slice(vertices);
            // Blender's `assignComponentTags` (intern/opensubdiv/internal/topology/
            // topology_refiner_factory.cc:238-244): a vertex with two edges adds the smaller of their
            // sharpnesses to its own (`crease_vert`), clamped to 10. It reads the edge sharpness set
            // from `crease_edge` before `derive_base_tags` makes boundary edges infinite.
            for (vertex, edges_at) in level.vertex_edges.iter().enumerate() {
                if let &[a, b] = edges_at.as_slice() {
                    let crease = level.edge_sharpness[a].min(level.edge_sharpness[b]);
                    level.vertex_sharpness[vertex] =
                        (level.vertex_sharpness[vertex] + crease).min(SHARPNESS_INFINITE);
                }
            }
            // Endpoints of loose edges are infinitely sharp (subdiv_converter_mesh.cc:346-353 sets
            // `infinite_sharp_vertices_map`; topology_refiner_factory.cc:221-225 applies it).
            for &[a, b] in *loose_edges {
                level.vertex_sharpness[a] = SHARPNESS_INFINITE;
                level.vertex_sharpness[b] = SHARPNESS_INFINITE;
            }
            (*preserve_corners, *non_manifold)
        }
        BaseSharpness::Simple => {
            level.edge_sharpness.fill(SHARPNESS_INFINITE);
            level.vertex_sharpness.fill(SHARPNESS_INFINITE);
            (false, no_non_manifold.as_slice())
        }
    };
    level.derive_base_tags(preserve_corners, non_manifold);
    Refined {
        level,
        positions: points.to_vec(),
        face_children: Vec::new(),
        dependent_source: Vec::new(),
        children: Children::default(),
    }
}

/// One uniform Catmull-Clark step of a base mesh (`use_limit_surface` off, or without the limit
/// surface's levels). The result is in the caller's numbering, so the caller can build the next step
/// from its own topology.
pub(super) struct UniformPoints {
    /// The child vertex of each input point on some face; a point on no face keeps its position.
    pub(super) vertices: Vec<DVec3>,
    /// The child vertex at the midpoint of each face edge, keyed by its sorted endpoints.
    pub(super) edges: BTreeMap<(usize, usize), DVec3>,
    /// The child vertex at the center of each face.
    pub(super) faces: Vec<DVec3>,
}

/// Refines every face of the base mesh once, with the masks of the refinement levels.
pub(super) fn uniform_points(
    polygons: &[Polygon],
    points: &[DVec3],
    sharpness: &BaseSharpness,
) -> UniformPoints {
    let base = base_level(polygons, points, sharpness);
    let selected = vec![true; polygons.len()];
    let child = refine(&base.level, &base.positions, &selected, 0);
    let vertices = points
        .iter()
        .enumerate()
        .map(|(vertex, &point)| {
            child.children.vertex[vertex].map_or(point, |index| child.positions[index])
        })
        .collect();
    let edges = base
        .level
        .edges
        .iter()
        .enumerate()
        .filter_map(|(edge, &[a, b])| {
            child.children.edge[edge].map(|index| (edge_key(a, b), child.positions[index]))
        })
        .collect();
    let faces = child
        .children
        .face
        .iter()
        .filter_map(|index| index.map(|index| child.positions[index]))
        .collect();
    UniformPoints {
        vertices,
        edges,
        faces,
    }
}

impl LimitSurface {
    /// Builds the refined levels of a base mesh to the isolation depth (Blender's quality).
    pub(super) fn build(
        polygons: &[Polygon],
        points: &[DVec3],
        sharpness: &BaseSharpness,
        isolation: usize,
    ) -> LimitSurface {
        let mut levels = vec![base_level(polygons, points, sharpness)];
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
        let ngon_root: Vec<bool> = polygons.iter().map(|(face, _)| face.len() != 4).collect();
        let has_irregular_faces = ngon_root.iter().any(|&irregular| irregular);
        LimitSurface {
            levels,
            selected,
            isolation,
            ngon_root,
            has_irregular_faces,
        }
    }

    /// The limit point of each sample (Blender's `evaluateLimit`): the leaf that holds the sample,
    /// evaluated at its local coordinates. A `None` sample stays `None`.
    pub(super) fn evaluate(&self, samples: &[Option<Sample>]) -> Vec<Option<DVec3>> {
        let mut slots = Vec::new();
        let mut leaves = Vec::new();
        for (slot, sample) in samples.iter().enumerate() {
            if let Some(sample) = *sample {
                slots.push(slot);
                leaves.push(locate(
                    &self.ngon_root,
                    &self.levels,
                    &self.selected,
                    sample,
                ));
            }
        }
        let values = evaluate_leaves(&self.levels, &leaves, self.has_irregular_faces, leaf_corner);
        let mut out = vec![None; samples.len()];
        for (slot, value) in slots.into_iter().zip(values) {
            out[slot] = Some(value);
        }
        out
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
        let (levels, selected) = self.face_varying_levels(linear, preserve_corners, map);
        let leaves: Vec<Leaf> = samples
            .corners
            .iter()
            .flatten()
            .map(|&sample| locate(&self.ngon_root, &levels, &selected, sample))
            .collect();
        if linear == FvarLinear::All {
            leaves
                .iter()
                .map(|leaf| {
                    let values = &levels[leaf.level];
                    bilinear(&values.level, &values.positions, leaf.face, leaf.s, leaf.t)
                })
                .collect()
        } else {
            evaluate_leaves(&levels, &leaves, self.has_irregular_faces, leaf_corner_fvar)
        }
    }

    /// The face-varying levels of a UV layer, and for each level the faces split into the next one.
    fn face_varying_levels(
        &self,
        linear: FvarLinear,
        preserve_corners: bool,
        map: &ValueMap,
    ) -> (Vec<Refined<DVec2>>, Vec<Vec<bool>>) {
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
            // Each value face's position face, through the parent/child links of both refinements.
            let mut positions: Vec<Option<usize>> =
                (0..levels[0].level.faces.len()).map(Some).collect();
            for depth in 0..self.isolation {
                let parent = &levels[depth];
                let extra: Vec<bool> = positions
                    .iter()
                    .map(|position| match (position, self.selected.get(depth)) {
                        (Some(position), Some(selection)) => selection[*position],
                        _ => false,
                    })
                    .collect();
                let Some((selection, child)) = refine_values_adaptive(
                    &parent.level,
                    &parent.positions,
                    depth,
                    false,
                    &parent.dependent_source,
                    &extra,
                ) else {
                    break;
                };
                positions = self.position_links(&child, &positions, depth);
                selected.push(selection);
                levels.push(child);
            }
        }
        (levels, selected)
    }

    /// The position face that each face of a value child covers at level `depth + 1`: the position child
    /// at the same corner of its value parent's position face, when the position refinement has it.
    fn position_links(
        &self,
        child: &Refined<DVec2>,
        parent: &[Option<usize>],
        depth: usize,
    ) -> Vec<Option<usize>> {
        debug_assert_eq!(parent.len(), child.face_children.len());
        let position_level = self.levels.get(depth + 1);
        let mut links = vec![None; child.level.faces.len()];
        for (value_parent, children) in child.face_children.iter().enumerate() {
            for (corner, &value_child) in children.iter().enumerate() {
                if let Some(value_child) = value_child {
                    links[value_child] = parent[value_parent].and_then(|position| {
                        position_level.and_then(|level| level.face_children[position][corner])
                    });
                }
            }
        }
        links
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
}

/// A sample's leaf: the level and face that hold it, and its local coordinates in that face.
#[derive(Clone, Copy)]
struct Leaf {
    level: usize,
    face: usize,
    s: f64,
    t: f64,
}

/// The leaf that holds `sample` (Blender's quadtree lookup) in a refinement with the given levels and
/// selections. Positions and face-varying values use the same lookup.
#[expect(
    clippy::expect_used,
    reason = "a sampled n-gon sub-face is refined at the first level, and every selected leaf has child quads"
)]
fn locate<T>(
    ngon_root: &[bool],
    levels: &[Refined<T>],
    selected: &[Vec<bool>],
    sample: Sample,
) -> Leaf {
    let (mut level, mut face, mut s, mut t) = if ngon_root[sample.face] {
        let child = levels[1].face_children[sample.face][sample.ptex]
            .expect("n-gon sub-faces are refined at the first level");
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
            .expect("a selected face has all of its child quads");
        (s, t) = child_coordinates(quadrant, s, t);
        face = child;
        level += 1;
    }
    Leaf { level, face, s, t }
}

/// Local coordinates of a point in child quad `quadrant` of its parent, in the child's own frame
/// (whose corner 0 is the parent's corner `quadrant`).
fn child_coordinates(quadrant: usize, s: f64, t: f64) -> (f64, f64) {
    match quadrant {
        0 => (2.0 * s, 2.0 * t),
        1 => (2.0 * t, 2.0 - 2.0 * s),
        2 => (2.0 - 2.0 * s, 2.0 - 2.0 * t),
        _ => (2.0 - 2.0 * t, 2.0 * s),
    }
}

/// The values at `leaves`, in their order. Leaves are visited grouped by level and face, so each
/// leaf's patch is built once.
fn evaluate_leaves<T: Point>(
    levels: &[Refined<T>],
    leaves: &[Leaf],
    has_irregular_faces: bool,
    corner: CornerBuilder,
) -> Vec<T> {
    let mut order: Vec<usize> = (0..leaves.len()).collect();
    order.sort_unstable_by_key(|&index| (leaves[index].level, leaves[index].face));
    let mut values = vec![T::default(); leaves.len()];
    let mut built: Option<((usize, usize), Patch)> = None;
    for index in order {
        let leaf = leaves[index];
        let key = (leaf.level, leaf.face);
        let patch = match built.take() {
            Some((built_key, patch)) if built_key == key => patch,
            _ => Patch::new(
                &levels[leaf.level].level,
                leaf.face,
                leaf.level,
                has_irregular_faces,
                corner,
            ),
        };
        values[index] = patch.evaluate(&levels[leaf.level].positions, leaf.s, leaf.t);
        built = Some((key, patch));
    }
    values
}

/// Corner coordinates of a quad sub-face, in corner order.
const QUAD_CORNERS: [(f64, f64); 4] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];

/// Grid samples of one level: the sample of each vertex that some face uses, and of each corner of
/// every face in the level's face order. A ptex coordinate is only meaningful in its own sub-face,
/// so midpoints and centers are averaged within one sub-face.
#[derive(Clone, Debug)]
pub(super) struct Samples {
    /// The sample of each vertex on some face; `None` for a vertex on no face.
    pub(super) vertices: Vec<Option<Sample>>,
    pub(super) corners: Vec<Vec<Sample>>,
}

impl Samples {
    /// Samples of the base mesh. A quad's corners are its ptex corners; an n-gon's corner `k` is the
    /// corner of its sub-face `k`, and its center is the `(1, 1)` corner of sub-face 0.
    pub(super) fn base(polygons: &[Polygon], vertex_count: usize) -> Samples {
        let mut vertices = vec![None; vertex_count];
        let mut corners = Vec::with_capacity(polygons.len());
        for (face, (corner_vertices, _)) in polygons.iter().enumerate() {
            let n = corner_vertices.len();
            let mut face_corners = Vec::with_capacity(n);
            for (k, &vertex) in corner_vertices.iter().enumerate() {
                let sample = if n == 4 {
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
    /// and each new face point, after the edge points, takes its face center. `polygons` are the
    /// current level's faces.
    pub(super) fn next(
        &self,
        polygons: &[Polygon],
        edge_points: &BTreeMap<(usize, usize), usize>,
        base_level: bool,
    ) -> Samples {
        let mut edge_samples = BTreeMap::new();
        for (face, (corner_vertices, _)) in polygons.iter().enumerate() {
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
        let mut vertices = vec![None; face_base + polygons.len()];
        vertices[..self.vertices.len()].copy_from_slice(&self.vertices);
        for (key, &index) in edge_points {
            vertices[index] = edge_samples.get(key).copied();
        }
        let mut children = Vec::with_capacity(self.corners.iter().map(Vec::len).sum());
        for (face, (corner_vertices, _)) in polygons.iter().enumerate() {
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
        u: f64::midpoint(a.u, b.u),
        v: f64::midpoint(a.v, b.v),
    }
}

/// Center of the corner samples of one sub-face.
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

/// Bilinear value of a linear channel's leaf quad at local coordinates (`OpenSubdiv`'s `QUADS` patches).
fn bilinear(level: &Level, positions: &[DVec2], face: usize, s: f64, t: f64) -> DVec2 {
    let weights = [(1.0 - s) * (1.0 - t), s * (1.0 - t), s * t, (1.0 - s) * t];
    level.faces[face]
        .iter()
        .zip(weights)
        .fold(DVec2::ZERO, |sum, (&value, weight)| {
            sum + positions[value] * weight
        })
}
