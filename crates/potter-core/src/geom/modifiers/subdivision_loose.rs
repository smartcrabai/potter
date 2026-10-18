// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/subdiv_mesh.cc, SPDX-FileCopyrightText: 2018 Blender Authors, GPL-2.0-or-later.

//! Loose-edge vertices of a subdivided mesh. Blender places each one from its base loose edge, not
//! from the level before: a point of the cubic B-spline through the edge's end points and their
//! neighbours on the base mesh (`find_loose_edge_interpolation_positions`). The parameter `u` runs
//! from the edge's first stored vertex, and the final level's vertices sit at `u = i / 2^levels`
//! (`subdiv_foreach_vertices_of_loose_edges_task`). The tracking keeps `u` for every loose vertex
//! through the levels, so the final positions depend on the base mesh only.

use std::collections::{BTreeMap, HashMap};

use glam::DVec3;

use crate::geom::edge_key;

/// A subdivided loose vertex: its base edge (an index into `LooseEdges::edges`) and `u` along it.
#[derive(Clone, Copy)]
struct Placement {
    edge: usize,
    u: f64,
}

/// The loose edges of the current level, with the placement of every loose vertex they created.
pub(super) struct LooseEdges {
    /// Every base edge in storage order, loose or on a face.
    edges: Vec<[usize; 2]>,
    /// The base edges at each base vertex (indices into `edges`).
    incident: Vec<Vec<usize>>,
    /// The current level's loose edges, keyed by their sorted endpoints. Each holds its base edge and
    /// the `u` of the smaller and of the larger endpoint.
    current: BTreeMap<(usize, usize), (usize, f64, f64)>,
    /// The placement of each vertex that a loose edge created, indexed by the current vertex.
    placements: Vec<Option<Placement>>,
}

impl LooseEdges {
    /// Tracks the loose edges of a base mesh with `vertex_count` vertices. `edges` lists all its
    /// edges and `loose` the loose ones, in the same stored orientation. `polygons` identifies
    /// endpoints whose face result must not be replaced by the loose-edge callback.
    pub(super) fn new(
        edges: Vec<[usize; 2]>,
        loose: &[[usize; 2]],
        polygons: &[super::Polygon],
        vertex_count: usize,
    ) -> Self {
        let mut incident = vec![Vec::new(); vertex_count];
        let mut index_of = HashMap::with_capacity(edges.len());
        for (index, &[first, second]) in edges.iter().enumerate() {
            incident[first].push(index);
            incident[second].push(index);
            index_of.insert(edge_key(first, second), index);
        }
        let mut face_vertices = vec![false; vertex_count];
        for (face, _) in polygons {
            for &vertex in face {
                face_vertices[vertex] = true;
            }
        }
        let mut current = BTreeMap::new();
        let mut placements = vec![None; vertex_count];
        for &[first, second] in loose {
            let key = edge_key(first, second);
            let Some(&index) = index_of.get(&key) else {
                continue;
            };
            // The stored first vertex has `u` 0 and the stored second has `u` 1.
            let (u_low, u_high) = if first == key.0 {
                (0.0, 1.0)
            } else {
                (1.0, 0.0)
            };
            current.insert(key, (index, u_low, u_high));
            if !face_vertices[first] {
                placements[first].get_or_insert(Placement {
                    edge: index,
                    u: 0.0,
                });
            }
            if !face_vertices[second] {
                placements[second].get_or_insert(Placement {
                    edge: index,
                    u: 1.0,
                });
            }
        }
        Self {
            edges,
            incident,
            current,
            placements,
        }
    }

    /// Moves to the next level. Each loose edge splits at the midpoint that `edge_points` gives it,
    /// and that midpoint's `u` is the mean of its ends. `vertex_count` is the new level's count.
    pub(super) fn split(
        &mut self,
        edge_points: &BTreeMap<(usize, usize), usize>,
        vertex_count: usize,
    ) {
        self.placements.resize(vertex_count, None);
        let mut next = BTreeMap::new();
        for (&(low, high), &(edge, u_low, u_high)) in &self.current {
            let Some(&middle) = edge_points.get(&(low, high)) else {
                continue;
            };
            let u_middle = f64::midpoint(u_low, u_high);
            self.placements[middle] = Some(Placement { edge, u: u_middle });
            let (first_low, first_high) = if low < middle {
                (u_low, u_middle)
            } else {
                (u_middle, u_low)
            };
            next.insert(edge_key(low, middle), (edge, first_low, first_high));
            let (second_low, second_high) = if middle < high {
                (u_middle, u_high)
            } else {
                (u_high, u_middle)
            };
            next.insert(edge_key(middle, high), (edge, second_low, second_high));
        }
        self.current = next;
    }

    /// The Blender output vertex index of each current loose-only vertex. Coarse endpoints retain
    /// their input indices; interior edge samples occupy their coarse-edge block.
    pub(super) fn blender_indices(
        &self,
        base_vertex_count: usize,
        resolution: usize,
    ) -> Vec<Option<usize>> {
        self.placements
            .iter()
            .map(|placement| {
                let placement = (*placement)?;
                let [first, second] = self.edges[placement.edge];
                let segment = (placement.u * (resolution - 1) as f64).round() as usize;
                Some(if segment == 0 {
                    first
                } else if segment == resolution - 1 {
                    second
                } else {
                    base_vertex_count + placement.edge * (resolution - 2) + segment - 1
                })
            })
            .collect()
    }

    /// The final position of every loose vertex, from the base points. Without `simple` each lies on
    /// the cubic B-spline through its base edge and the neighbouring base edges; with it, on the
    /// straight edge (`mesh_interpolate_position_on_edge`).
    pub(super) fn positions(&self, points: &[DVec3], simple: bool) -> Vec<(usize, DVec3)> {
        self.placements
            .iter()
            .enumerate()
            .filter_map(|(vertex, placement)| {
                let placement = (*placement)?;
                let [first, second] = self.edges[placement.edge];
                let u = placement.u;
                let position = if simple {
                    points[first] * (1.0 - u) + points[second] * u
                } else {
                    let p1 = points[first];
                    let p2 = points[second];
                    let p0 = self
                        .neighbour(placement.edge, first)
                        .map_or(p1 * 2.0 - p2, |vertex| points[vertex]);
                    let p3 = self
                        .neighbour(placement.edge, second)
                        .map_or(p2 * 2.0 - p1, |vertex| points[vertex]);
                    let weights = bspline_weights(u);
                    p0 * weights[0] + p1 * weights[1] + p2 * weights[2] + p3 * weights[3]
                };
                Some((vertex, position))
            })
            .collect()
    }

    /// The base vertices a loose-only vertex blends, with Blender's weights, or `None` without a
    /// placement. A loose end point copies its base vertex. An interior point mixes the stored ends of
    /// its base edge by `(1 - u, u)` (`subdiv_mesh_vert_of_loose_edge_interpolate`, `subdiv_mesh.cc:1434`).
    pub(super) fn sources(&self, vertex: usize) -> Option<Vec<(usize, f64)>> {
        let placement = (*self.placements.get(vertex)?)?;
        let [first, second] = self.edges[placement.edge];
        Some(if placement.u <= 0.0 {
            vec![(first, 1.0)]
        } else if placement.u >= 1.0 {
            vec![(second, 1.0)]
        } else {
            vec![(first, 1.0 - placement.u), (second, placement.u)]
        })
    }

    /// The other end of the base edge that continues `edge` at `vertex`. A vertex with exactly two
    /// edges has one (`find_edge_neighbors`); any other vertex has none.
    fn neighbour(&self, edge: usize, vertex: usize) -> Option<usize> {
        let at = &self.incident[vertex];
        if at.len() != 2 {
            return None;
        }
        let other = if at[0] == edge { at[1] } else { at[0] };
        let [first, second] = self.edges[other];
        Some(if first == vertex { second } else { first })
    }
}

/// `key_curve_position_weights` for `KEY_BSPLINE` (Blender `key.cc`), with its single-precision
/// constants.
fn bspline_weights(u: f64) -> [f64; 4] {
    let u2 = u * u;
    let u3 = u2 * u;
    [
        -0.166_666_66 * u3 + 0.5 * u2 - 0.5 * u + 0.166_666_66,
        0.5 * u3 - u2 + 0.666_666_66,
        -0.5 * u3 + 0.5 * u2 + 0.5 * u + 0.166_666_66,
        0.166_666_66 * u3,
    ]
}
