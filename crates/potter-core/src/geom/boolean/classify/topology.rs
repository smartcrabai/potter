//! Edge and vertex incidence of an intersected triangle mesh (Blender `TriMeshTopology`) and the
//! PWN test (`is_pwn`).

use std::collections::HashMap;

use super::super::imesh::Tri;

/// Undirected edge between two arena vertices, stored with `v0 <= v1`. Blender orders by vertex
/// id, which is the arena index here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Edge {
    pub(super) v0: u32,
    pub(super) v1: u32,
}

impl Edge {
    #[must_use]
    pub(super) fn new(a: u32, b: u32) -> Self {
        if a <= b {
            Self { v0: a, v1: b }
        } else {
            Self { v0: b, v1: a }
        }
    }

    /// The endpoint of this edge that is not `v`.
    #[must_use]
    pub(super) fn other(self, v: u32) -> u32 {
        if self.v0 == v { self.v1 } else { self.v0 }
    }
}

/// Triangles incident on each edge and edges incident on each vertex.
#[derive(Debug, Default)]
pub(super) struct Topology {
    edge_tris: HashMap<Edge, Vec<usize>>,
    vert_edges: HashMap<u32, Vec<Edge>>,
}

impl Topology {
    #[must_use]
    pub(super) fn new(tris: &[Tri]) -> Self {
        let mut edge_tris: HashMap<Edge, Vec<usize>> = HashMap::new();
        let mut vert_edges: HashMap<u32, Vec<Edge>> = HashMap::new();
        for (t, tri) in tris.iter().enumerate() {
            for (i, &v) in tri.v.iter().enumerate() {
                let e = Edge::new(v, tri.v[(i + 1) % 3]);
                let edges = vert_edges.entry(v).or_default();
                if !edges.contains(&e) {
                    edges.push(e);
                }
                let incident = edge_tris.entry(e).or_default();
                if !incident.contains(&t) {
                    incident.push(t);
                }
            }
        }
        Self {
            edge_tris,
            vert_edges,
        }
    }

    /// Blender `other_tri_if_manifold`: the other triangle of a two-triangle edge, else `None`.
    #[must_use]
    pub(super) fn other_tri_if_manifold(&self, e: Edge, t: usize) -> Option<usize> {
        let incident = self.edge_tris.get(&e)?;
        match incident.as_slice() {
            [a, b] => Some(if *a == t { *b } else { *a }),
            _ => None,
        }
    }

    /// Triangles that contain edge `e` in either orientation.
    #[must_use]
    pub(super) fn edge_tris(&self, e: Edge) -> Option<&[usize]> {
        self.edge_tris.get(&e).map(Vec::as_slice)
    }

    /// Edges incident on vertex `v`; empty if `v` is unused.
    #[must_use]
    pub(super) fn vert_edges(&self, v: u32) -> &[Edge] {
        self.vert_edges
            .get(&v)
            .map_or_else(Default::default, Vec::as_slice)
    }

    fn all_edge_tris(&self) -> impl Iterator<Item = (Edge, &[usize])> {
        self.edge_tris.iter().map(|(&e, ts)| (e, ts.as_slice()))
    }
}

/// +1 if `tri` contains the directed edge `e.v0 -> e.v1`, -1 if it contains `e.v1 -> e.v0`.
fn directed_sign(tri: &Tri, e: Edge) -> i32 {
    let mut total = 0;
    for (i, &v) in tri.v.iter().enumerate() {
        if v == e.v0 {
            if tri.v[(i + 1) % 3] == e.v1 {
                total += 1;
            } else {
                total -= 1;
            }
        }
    }
    total
}

/// Blender `is_pwn`: every edge must have zero net orientation over its incident triangles.
#[must_use]
pub(super) fn is_pwn(tris: &[Tri], topo: &Topology) -> bool {
    topo.all_edge_tris().all(|(e, incident)| {
        let total: i32 = incident.iter().map(|&t| directed_sign(&tris[t], e)).sum();
        total == 0
    })
}

/// Blender `find_flap_vert`: if `e` is an edge of `tri`, the vertex of `tri` not on `e`, and
/// whether `tri` traverses `e` reversed (`v1 -> v0`).
#[must_use]
pub(super) fn find_flap_vert(tri: &Tri, e: Edge) -> Option<(u32, bool)> {
    let [t0, t1, t2] = tri.v;
    if t0 == e.v0 {
        if t1 == e.v1 {
            Some((t2, false))
        } else if t2 == e.v1 {
            Some((t1, true))
        } else {
            None
        }
    } else if t1 == e.v0 {
        if t2 == e.v1 {
            Some((t0, false))
        } else if t0 == e.v1 {
            Some((t2, true))
        } else {
            None
        }
    } else if t2 == e.v0 {
        if t0 == e.v1 {
            Some((t1, false))
        } else if t1 == e.v1 {
            Some((t0, true))
        } else {
            None
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri(v: [u32; 3]) -> Tri {
        Tri::new(v, 0, [-1; 3])
    }

    #[test]
    fn flap_vertex_and_reversal() {
        let e = Edge::new(1, 2);
        // Forward: triangle has 1 -> 2, flap is 3.
        assert_eq!(find_flap_vert(&tri([1, 2, 3]), e), Some((3, false)));
        // Reversed: triangle has 2 -> 1, flap is 3.
        assert_eq!(find_flap_vert(&tri([2, 1, 3]), e), Some((3, true)));
        // Edge absent.
        assert_eq!(find_flap_vert(&tri([1, 4, 5]), e), None);
    }

    #[test]
    fn closed_tetrahedron_is_pwn_and_open_triangle_is_not() {
        let closed = [
            tri([0, 2, 1]),
            tri([0, 1, 3]),
            tri([1, 2, 3]),
            tri([2, 0, 3]),
        ];
        assert!(is_pwn(&closed, &Topology::new(&closed)));
        let open = [tri([0, 1, 2])];
        assert!(!is_pwn(&open, &Topology::new(&open)));
    }
}
