// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   intern/dualcon/intern/octree.cpp (cellProcParity/faceProcParity/edgeProcParity and
//   cellProcContour/faceProcContour/edgeProcContour), SPDX-FileCopyrightText: 2011-2023 Blender
//   Authors, GPL-2.0-or-later.

//! Cell, face, and edge recursion over the octree (Blender's `*Proc*` traversals).
//!
//! Blender runs two traversals with identical recursion: the parity pass that unifies minimal
//! edges (`resetMinimalEdges`) and the contour pass that emits quads. Only the leaf-level edge action
//! differs, so both share this walker.

use super::{
    octree::{Child, Octree, child_slots},
    tables::{
        CELL_PROC_EDGE_MASK, CELL_PROC_FACE_MASK, EDGE_PROC_EDGE_MASK, FACE_PROC_EDGE_MASK,
        FACE_PROC_FACE_MASK,
    },
};
use crate::error::Result;

/// Which leaf-level action an edge traversal performs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Pass {
    /// `processEdgeParity`: propagate minimal-edge parity across the four leaves.
    Parity,
    /// `processEdgeWrite`: emit the quad for an intersecting primary edge.
    Contour,
}

impl Octree {
    /// `cellProcParity` / `cellProcContour`. Leaves are visited through their parent only.
    pub(super) fn walk_cell(&mut self, node: Child, pass: Pass) -> Result<()> {
        let Child::Internal(id) = node else {
            return Ok(());
        };
        let children = self.internals[id as usize].children;
        for child in children {
            self.walk_cell(child, pass)?;
        }
        for mask in CELL_PROC_FACE_MASK {
            self.walk_face([children[mask[0]], children[mask[1]]], mask[2], pass)?;
        }
        for mask in CELL_PROC_EDGE_MASK {
            self.walk_edge(
                [
                    children[mask[0]],
                    children[mask[1]],
                    children[mask[2]],
                    children[mask[3]],
                ],
                mask[4],
                pass,
            )?;
        }
        Ok(())
    }

    /// `faceProcParity` / `faceProcContour`.
    fn walk_face(&mut self, nodes: [Child; 2], direction: usize, pass: Pass) -> Result<()> {
        if nodes.contains(&Child::Empty) {
            return Ok(());
        }
        if nodes.iter().all(|node| matches!(node, Child::Leaf(_))) {
            return Ok(());
        }
        let children = nodes.map(|node| child_slots(&self.internals, node));
        for face in FACE_PROC_FACE_MASK[direction] {
            let next: [Child; 2] = std::array::from_fn(|side| {
                if matches!(nodes[side], Child::Leaf(_)) {
                    nodes[side]
                } else {
                    children[side][face[side]]
                }
            });
            self.walk_face(next, face[2], pass)?;
        }
        let orders = [[0, 0, 1, 1], [0, 1, 0, 1]];
        for edge in FACE_PROC_EDGE_MASK[direction] {
            let order = orders[edge[0]];
            let next: [Child; 4] = std::array::from_fn(|slot| {
                let side = order[slot];
                if matches!(nodes[side], Child::Leaf(_)) {
                    nodes[side]
                } else {
                    children[side][edge[slot + 1]]
                }
            });
            self.walk_edge(next, edge[5], pass)?;
        }
        Ok(())
    }

    /// `edgeProcParity` / `edgeProcContour`.
    fn walk_edge(&mut self, nodes: [Child; 4], direction: usize, pass: Pass) -> Result<()> {
        if nodes.contains(&Child::Empty) {
            return Ok(());
        }
        if nodes.iter().all(|node| matches!(node, Child::Leaf(_))) {
            let leaves = nodes.map(|node| match node {
                Child::Leaf(leaf) => leaf,
                _ => unreachable!("all nodes were checked to be leaves"),
            });
            match pass {
                Pass::Parity => self.process_edge_parity(leaves, direction),
                Pass::Contour => self.process_edge_write(leaves, direction)?,
            }
            return Ok(());
        }
        let children = nodes.map(|node| child_slots(&self.internals, node));
        for edge in EDGE_PROC_EDGE_MASK[direction] {
            let next: [Child; 4] = std::array::from_fn(|side| {
                if matches!(nodes[side], Child::Leaf(_)) {
                    nodes[side]
                } else {
                    children[side][edge[side]]
                }
            });
            self.walk_edge(next, edge[4], pass)?;
        }
        Ok(())
    }
}
