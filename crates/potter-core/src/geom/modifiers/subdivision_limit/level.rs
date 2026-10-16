// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from OpenSubdiv (https://graphics.pixar.com/opensubdiv) vtr/level.h, far/topologyRefinerFactory.cpp, sdc/crease.h, Apache-2.0.

//! One Catmull-Clark topology level with OpenSubdiv's per-component tags.

use std::collections::BTreeMap;

/// Sharpness at or above this value is infinitely sharp (OpenSubdiv `SHARPNESS_INFINITE`).
pub(super) const SHARPNESS_INFINITE: f64 = 10.0;

/// Vertex rule bits, combined by bitwise OR over a face's corners.
pub(super) const RULE_SMOOTH: u8 = 1;
pub(super) const RULE_DART: u8 = 2;
pub(super) const RULE_CREASE: u8 = 4;
pub(super) const RULE_CORNER: u8 = 8;

/// Per-vertex tags (OpenSubdiv `Vtr::Level::VTag`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct VTag {
    pub(super) non_manifold: bool,
    pub(super) xordinary: bool,
    pub(super) boundary: bool,
    pub(super) corner: bool,
    pub(super) inf_sharp: bool,
    pub(super) semi_sharp: bool,
    pub(super) semi_sharp_edges: bool,
    pub(super) incomplete: bool,
    pub(super) incid_irreg_face: bool,
    pub(super) inf_sharp_edges: bool,
    pub(super) inf_sharp_crease: bool,
    pub(super) inf_irregular: bool,
    pub(super) rule: u8,
}

impl VTag {
    /// Bitwise OR of the tags of a set of corners (OpenSubdiv `VTag::BitwiseOr`).
    pub(super) fn combine<'a>(tags: impl IntoIterator<Item = &'a VTag>) -> VTag {
        tags.into_iter().fold(VTag::default(), |mut acc, tag| {
            acc.non_manifold |= tag.non_manifold;
            acc.xordinary |= tag.xordinary;
            acc.boundary |= tag.boundary;
            acc.corner |= tag.corner;
            acc.inf_sharp |= tag.inf_sharp;
            acc.semi_sharp |= tag.semi_sharp;
            acc.semi_sharp_edges |= tag.semi_sharp_edges;
            acc.incomplete |= tag.incomplete;
            acc.incid_irreg_face |= tag.incid_irreg_face;
            acc.inf_sharp_edges |= tag.inf_sharp_edges;
            acc.inf_sharp_crease |= tag.inf_sharp_crease;
            acc.inf_irregular |= tag.inf_irregular;
            acc.rule |= tag.rule;
            acc
        })
    }
}

/// Per-edge tags (OpenSubdiv `Vtr::Level::ETag`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ETag {
    pub(super) non_manifold: bool,
    pub(super) boundary: bool,
    pub(super) inf_sharp: bool,
    pub(super) semi_sharp: bool,
    /// The fvar edge has linear interpolation rather than a smooth boundary crease.
    pub(super) linear: bool,
}

/// Sharpness is infinite.
pub(super) fn is_infinite(sharpness: f64) -> bool {
    sharpness >= SHARPNESS_INFINITE
}

/// Sharpness is strictly positive.
pub(super) fn is_sharp(sharpness: f64) -> bool {
    sharpness > 0.0
}

/// Sharpness is positive but not infinite.
pub(super) fn is_semi_sharp(sharpness: f64) -> bool {
    sharpness > 0.0 && sharpness < SHARPNESS_INFINITE
}

/// Uniform crease decay per level (OpenSubdiv `Crease::decrementSharpness`).
pub(super) fn decrement(sharpness: f64) -> f64 {
    if sharpness <= 0.0 {
        0.0
    } else if sharpness >= SHARPNESS_INFINITE {
        SHARPNESS_INFINITE
    } else if sharpness > 1.0 {
        sharpness - 1.0
    } else {
        0.0
    }
}

/// Vertex rule from its own sharpness and its count of sharp incident edges
/// (OpenSubdiv `Crease::DetermineVertexVertexRule`).
pub(super) fn vertex_rule(sharpness: f64, sharp_edges: usize) -> u8 {
    if is_sharp(sharpness) || sharp_edges > 2 {
        RULE_CORNER
    } else if sharp_edges == 0 {
        RULE_SMOOTH
    } else if sharp_edges == 1 {
        RULE_DART
    } else {
        RULE_CREASE
    }
}

/// A quad-split Catmull-Clark topology level. Faces list their corners in order; `face_edges[f][k]`
/// joins corner `k` to corner `k + 1`.
#[derive(Clone, Debug, Default)]
pub(super) struct Level {
    pub(super) faces: Vec<Vec<usize>>,
    pub(super) face_edges: Vec<Vec<usize>>,
    pub(super) edges: Vec<[usize; 2]>,
    pub(super) edge_faces: Vec<Vec<usize>>,
    pub(super) vertex_faces: Vec<Vec<usize>>,
    pub(super) vertex_edges: Vec<Vec<usize>>,
    pub(super) edge_sharpness: Vec<f64>,
    pub(super) vertex_sharpness: Vec<f64>,
    pub(super) vtags: Vec<VTag>,
    pub(super) etags: Vec<ETag>,
}

impl Level {
    /// Builds the topology from corner lists. Edges are keyed by their sorted endpoints.
    pub(super) fn from_faces(faces: Vec<Vec<usize>>, vertex_count: usize) -> Level {
        let mut edge_index = BTreeMap::<(usize, usize), usize>::new();
        let mut edges = Vec::new();
        let mut face_edges = Vec::with_capacity(faces.len());
        for face in &faces {
            let mut ids = Vec::with_capacity(face.len());
            for k in 0..face.len() {
                let a = face[k];
                let b = face[(k + 1) % face.len()];
                let key = (a.min(b), a.max(b));
                let id = *edge_index.entry(key).or_insert_with(|| {
                    edges.push([key.0, key.1]);
                    edges.len() - 1
                });
                ids.push(id);
            }
            face_edges.push(ids);
        }
        Level::from_parts(faces, face_edges, edges, vertex_count)
    }

    /// Builds the incidence from corners, their edges, and the edge endpoints. Tags and sharpness
    /// start cleared; callers set them.
    pub(super) fn from_parts(
        faces: Vec<Vec<usize>>,
        face_edges: Vec<Vec<usize>>,
        edges: Vec<[usize; 2]>,
        vertex_count: usize,
    ) -> Level {
        let mut edge_faces = vec![Vec::new(); edges.len()];
        let mut vertex_faces = vec![Vec::new(); vertex_count];
        let mut vertex_edges = vec![Vec::new(); vertex_count];
        for (f, (face, ids)) in faces.iter().zip(&face_edges).enumerate() {
            let n = face.len();
            for (corner, &v) in face.iter().enumerate() {
                vertex_faces[v].push(f);
                for e in [ids[corner], ids[(corner + n - 1) % n]] {
                    if !vertex_edges[v].contains(&e) {
                        vertex_edges[v].push(e);
                    }
                }
            }
            for &e in ids {
                edge_faces[e].push(f);
            }
        }
        let edge_count = edges.len();
        Level {
            faces,
            face_edges,
            edges,
            edge_faces,
            vertex_faces,
            vertex_edges,
            edge_sharpness: vec![0.0; edge_count],
            vertex_sharpness: vec![0.0; vertex_count],
            vtags: vec![VTag::default(); vertex_count],
            etags: vec![ETag::default(); edge_count],
        }
    }


    pub(super) fn vertex_count(&self) -> usize {
        self.vertex_faces.len()
    }

    pub(super) fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Corner index of `vertex` within `face`.
    pub(super) fn corner_of(&self, face: usize, vertex: usize) -> usize {
        self.faces[face]
            .iter()
            .position(|&corner| corner == vertex)
            .unwrap_or(0)
    }

    /// The face on the other side of `edge` from `face`, if the edge is manifold and interior.
    pub(super) fn other_face(&self, edge: usize, face: usize) -> Option<usize> {
        self.edge_faces[edge].iter().copied().find(|&other| other != face)
    }

    /// Far endpoint of `edge` relative to `vertex`.
    pub(super) fn far_vertex(&self, edge: usize, vertex: usize) -> usize {
        let [a, b] = self.edges[edge];
        if a == vertex { b } else { a }
    }

    /// Base-level tags, following OpenSubdiv's `prepareComponentTagsAndSharpness`.
    pub(super) fn derive_base_tags(
        &mut self,
        preserve_corners: bool,
        non_manifold_vertices: &[bool],
    ) {
        let manifold_sizes_regular = self.faces.iter().all(|face| face.len() == 4);
        for edge in 0..self.edges.len() {
            let boundary = self.edge_faces[edge].len() < 2;
            let non_manifold = self.edge_faces[edge].len() > 2;
            if boundary || non_manifold {
                self.edge_sharpness[edge] = SHARPNESS_INFINITE;
            }
            self.etags[edge] = ETag {
                non_manifold,
                boundary,
                inf_sharp: is_infinite(self.edge_sharpness[edge]),
                semi_sharp: is_semi_sharp(self.edge_sharpness[edge]),
                linear: false,
            };
        }
        for vertex in 0..self.vertex_count() {
            let edges = self.vertex_edges[vertex].clone();
            let faces = self.vertex_faces[vertex].len();
            let boundary_count = edges.iter().filter(|&&e| self.etags[e].boundary).count();
            let inf_count = edges.iter().filter(|&&e| self.etags[e].inf_sharp).count();
            let semi_count = edges.iter().filter(|&&e| self.etags[e].semi_sharp).count();
            let non_manifold = non_manifold_vertices[vertex];
            let topological_corner = faces == 1 && edges.len() == 2;
            if topological_corner && preserve_corners {
                self.vertex_sharpness[vertex] = SHARPNESS_INFINITE;
            } else if non_manifold && !is_infinite(self.vertex_sharpness[vertex]) {
                self.vertex_sharpness[vertex] = SHARPNESS_INFINITE;
            }
            let sharpness = self.vertex_sharpness[vertex];
            let sharp_edges = inf_count + semi_count;
            let corner = topological_corner && is_infinite(sharpness);
            let xordinary = if non_manifold || corner {
                false
            } else if boundary_count > 0 {
                faces != 2
            } else {
                faces != 4
            };
            let inf_infinite = is_infinite(sharpness);
            let inf_rule = vertex_rule(if inf_infinite { sharpness } else { 0.0 }, inf_count);
            let mut inf_crease = false;
            let mut inf_irregular = inf_infinite || inf_count > 0;
            if inf_count > 0 {
                if inf_rule == RULE_CREASE {
                    inf_crease = true;
                    if !xordinary && !non_manifold {
                        if boundary_count > 0 {
                            inf_irregular = false;
                        } else if edges.len() == 4 {
                            inf_irregular = self.etags[edges[0]].inf_sharp
                                != self.etags[edges[2]].inf_sharp;
                        }
                    }
                } else if inf_rule == RULE_CORNER
                    && inf_count == edges.len()
                    && (edges.len() > 2 || inf_infinite)
                {
                    inf_irregular = false;
                }
            }
            let incid_irreg_face = !manifold_sizes_regular
                && self.vertex_faces[vertex]
                    .iter()
                    .any(|&f| self.faces[f].len() != 4);
            self.vtags[vertex] = VTag {
                non_manifold,
                xordinary,
                boundary: boundary_count > 0,
                corner,
                inf_sharp: inf_infinite,
                semi_sharp: is_semi_sharp(sharpness),
                semi_sharp_edges: semi_count > 0,
                incomplete: false,
                incid_irreg_face,
                inf_sharp_edges: inf_count > 0,
                inf_sharp_crease: inf_crease,
                inf_irregular,
                rule: vertex_rule(sharpness, sharp_edges),
            };
        }
    }
}
