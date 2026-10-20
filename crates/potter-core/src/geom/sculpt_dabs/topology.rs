//! Mesh topology for the dab brushes, built once per stroke. Pose and Boundary visit neighbors in
//! `ring` order, and the order decides which vertex claims a tie, so it is kept exactly. Smooth
//! averages over `interior`.

use std::collections::BTreeMap;

use crate::geom::Mesh;

/// Blender's neighbor and boundary data for one mesh, indexed by vertex position in
/// `mesh.vertices`.
pub(super) struct Topology {
    /// `vert_neighbors_get_mesh`: for each face that uses the vertex, in face order, the face's
    /// previous corner then its next corner, without duplicates.
    pub(super) ring: Vec<Vec<usize>>,
    /// `create_boundary_info`: whether the vertex is an endpoint of an edge used by fewer than two
    /// faces. Loose edges count as boundary.
    pub(super) boundary: Vec<bool>,
    /// `calc_vert_neighbors_interior` (`sculpt.cc` 8092): `ring` without the neighbors a boundary
    /// vertex must not average over. A boundary vertex keeps only neighbors joined by a boundary
    /// edge, and none when it has exactly two neighbors (a corner). Sorted.
    pub(super) interior: Vec<Vec<usize>>,
}

impl Topology {
    pub(super) fn new(mesh: &Mesh, index: &BTreeMap<u32, usize>) -> Self {
        let count = mesh.vertices.len();
        let mut ring = vec![Vec::new(); count];
        let mut face_uses = BTreeMap::<(u32, u32), usize>::new();
        for face in &mesh.faces {
            let corners = &face.vertices;
            let corner_count = corners.len();
            for corner in 0..corner_count {
                let vertex = corners[corner];
                let next = corners[(corner + 1) % corner_count];
                let previous = corners[(corner + corner_count - 1) % corner_count];
                *face_uses.entry(edge_key(vertex, next)).or_insert(0) += 1;
                if let Some(&position) = index.get(&vertex) {
                    for neighbor in [previous, next] {
                        if let Some(&neighbor) = index.get(&neighbor)
                            && !ring[position].contains(&neighbor)
                        {
                            ring[position].push(neighbor);
                        }
                    }
                }
            }
        }
        let mut boundary = vec![false; count];
        for edge in &mesh.edges {
            let uses = face_uses
                .get(&edge_key(edge.vertices[0], edge.vertices[1]))
                .copied()
                .unwrap_or(0);
            if uses < 2 {
                for id in edge.vertices {
                    if let Some(&position) = index.get(&id) {
                        boundary[position] = true;
                    }
                }
            }
        }
        let interior = (0..count)
            .map(|position| {
                let around = &ring[position];
                let mut list: Vec<usize> = if !boundary[position] {
                    around.clone()
                } else if around.len() == 2 {
                    Vec::new()
                } else {
                    let id = mesh.vertices[position].id;
                    around
                        .iter()
                        .copied()
                        .filter(|&neighbor| {
                            let neighbor_id = mesh.vertices[neighbor].id;
                            face_uses
                                .get(&edge_key(id, neighbor_id))
                                .is_some_and(|&uses| uses < 2)
                        })
                        .collect()
                };
                list.sort_unstable();
                list
            })
            .collect();
        Self {
            ring,
            boundary,
            interior,
        }
    }
}

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}
