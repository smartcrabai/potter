//! Mesh topology in Blender's neighbor order, for the Pose and Boundary brushes. Their flood fills
//! and propagations visit neighbors in this order, and the order decides which vertex claims a
//! tie, so it is kept exactly.

use std::collections::BTreeMap;

use crate::geom::Mesh;

/// Blender's `vert_neighbors_get_mesh` and `create_boundary_info` for one mesh, indexed by vertex
/// position in `mesh.vertices`.
pub(super) struct Topology {
    /// `vert_neighbors_get_mesh`: for each face that uses the vertex, in face order, the face's
    /// previous corner then its next corner, without duplicates.
    pub(super) ring: Vec<Vec<usize>>,
    /// `create_boundary_info`: whether the vertex is an endpoint of an edge used by fewer than two
    /// faces. Loose edges count as boundary.
    pub(super) boundary: Vec<bool>,
}

impl Topology {
    pub(super) fn new(mesh: &Mesh, index: &BTreeMap<u32, usize>) -> Self {
        let mut ring = vec![Vec::new(); mesh.vertices.len()];
        let mut face_uses = BTreeMap::<(u32, u32), usize>::new();
        for face in &mesh.faces {
            let corners = &face.vertices;
            let count = corners.len();
            for corner in 0..count {
                let vertex = corners[corner];
                let next = corners[(corner + 1) % count];
                let previous = corners[(corner + count - 1) % count];
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
        let mut boundary = vec![false; mesh.vertices.len()];
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
        Self { ring, boundary }
    }
}

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}
