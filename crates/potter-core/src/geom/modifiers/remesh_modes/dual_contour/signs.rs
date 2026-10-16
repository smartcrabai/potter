// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   intern/dualcon/intern/octree.cpp (buildSigns, floodFill, clearProcessBits), SPDX-FileCopyrightText:
//   2011-2023 Blender Authors, GPL-2.0-or-later.

//! Corner-sign generation from edge parity, and the two-pass component filter (`use_remove_disconnected`).

use std::collections::VecDeque;

use super::{
    octree::{Child, Octree, offset_position},
    tables::{EDGEMAP, FLOOD_FC_CELLS, FLOOD_FC_EDGES, FLOOD_ST_MASK, VERTMAP},
};
use crate::error::Result;

/// Parity-to-sign lookup. For every edge-parity pattern, the sign mask whose parity matches; ties
/// resolve to the later mask, exactly as Blender's `buildSigns` table loop does.
fn sign_table() -> Vec<u8> {
    let mut table = vec![0_u8; 1 << 12];
    for mask in 0..256_u32 {
        let mut index = 0_usize;
        for edge in (0..12).rev() {
            index <<= 1;
            let [first, second] = EDGEMAP[edge];
            if ((mask >> first) & 1) ^ ((mask >> second) & 1) == 1 {
                index |= 1;
            }
        }
        table[index] = mask as u8;
    }
    table
}

impl Octree {
    /// `buildSigns()`: assigns corner signs to every leaf, starting from sign 1 at the root.
    pub(super) fn build_signs(&mut self) {
        let table = sign_table();
        let mut root_signs = [0_u8; 8];
        let root = Child::Internal(Self::root());
        self.build_signs_node(&table, root, 1, &mut root_signs);
    }

    /// `buildSigns(table, node, isLeaf, sg, rvalue)`. `rvalue` receives the corner signs of `node`.
    fn build_signs_node(&mut self, table: &[u8], node: Child, start: u8, rvalue: &mut [u8; 8]) {
        match node {
            Child::Empty => *rvalue = [start; 8],
            Child::Internal(id) => {
                let children = self.internals[id as usize].children;
                rvalue[0] = start;
                let mut first_child = [0_u8; 8];
                self.build_signs_node(table, children[0], start, &mut first_child);
                for octant in 1..8 {
                    let mut child_signs = [0_u8; 8];
                    self.build_signs_node(
                        table,
                        children[octant],
                        first_child[octant],
                        &mut child_signs,
                    );
                    rvalue[octant] = child_signs[octant];
                }
            }
            Child::Leaf(leaf) => {
                let parity = usize::from(self.leaf(leaf).edge_parity);
                let mut signs = table[parity];
                if (start ^ signs) & 1 == 1 {
                    signs = !signs;
                }
                self.leaf_mut(leaf).signs = signs;
                for (octant, value) in rvalue.iter_mut().enumerate() {
                    *value = (signs >> octant) & 1;
                }
            }
        }
    }

    /// `floodFill()`: removes intersection components smaller than `thresh` times the largest one.
    pub(super) fn flood_fill(&mut self) -> Result<()> {
        let root = Child::Internal(Self::root());
        let dimen = self.dimen;

        self.clear_process_bits(root);
        let largest = self.flood_fill_node(root, [0; 3], dimen, 0)?;

        let threshold = ((largest as f32) * self.thresh) as i32;
        self.clear_process_bits(root);
        self.flood_fill_node(root, [0; 3], dimen, threshold)?;
        Ok(())
    }

    /// `clearProcessBits`.
    fn clear_process_bits(&mut self, node: Child) {
        match node {
            Child::Empty => {}
            Child::Leaf(leaf) => self.leaf_mut(leaf).flood_fill = 0,
            Child::Internal(id) => {
                let children = self.internals[id as usize].children;
                for child in children {
                    self.clear_process_bits(child);
                }
            }
        }
    }

    /// `floodFill(Node*, ...)`: returns the largest component size when `threshold == 0`.
    fn flood_fill_node(
        &mut self,
        node: Child,
        st: [i32; 3],
        len: i32,
        threshold: i32,
    ) -> Result<i32> {
        match node {
            Child::Empty => Ok(0),
            Child::Leaf(leaf) => self.flood_fill_leaf(leaf, st, len, threshold),
            Child::Internal(id) => {
                let children = self.internals[id as usize].children;
                let len = len >> 1;
                let mut maximum = 0;
                for (octant, child) in children.iter().enumerate() {
                    if *child == Child::Empty {
                        continue;
                    }
                    let child_origin = offset_position(st, VERTMAP[octant], len);
                    let total = self.flood_fill_node(*child, child_origin, len, threshold)?;
                    maximum = maximum.max(total);
                }
                Ok(maximum)
            }
        }
    }

    /// The first (sign 0: forward; otherwise reverse) neighbour edge on a face, selected by parity
    /// or by the flood-fill in-process bit.
    fn neighbour_edge(
        &self,
        leaf: u32,
        candidates: [usize; 3],
        sign: u8,
        in_process: bool,
    ) -> Option<usize> {
        let matches = |edge: usize| {
            if in_process {
                (self.leaf(leaf).flood_fill >> edge) & 1 == 1
            } else {
                self.parity(leaf, edge)
            }
        };
        if sign == 0 {
            candidates.into_iter().find(|edge| matches(*edge))
        } else {
            candidates.into_iter().rev().find(|edge| matches(*edge))
        }
    }

    /// `floodFill(LeafNode*, ...)`: walks each unvisited intersection edge of one leaf. The first
    /// pass measures the component (`threshold == 0`); the second removes it when it is too small.
    fn flood_fill_leaf(
        &mut self,
        leaf: u32,
        st: [i32; 3],
        len: i32,
        threshold: i32,
    ) -> Result<i32> {
        let mut maximum = 0;
        for edge in 0..12 {
            let in_process = (self.leaf(leaf).flood_fill >> edge) & 1 == 1;
            if !self.parity(leaf, edge) || in_process {
                continue;
            }

            let mut total = 1_i32;
            let mut queue = VecDeque::new();
            let start = offset_position(st, VERTMAP[EDGEMAP[edge][0]], len);
            let direction = edge / 4;
            self.set_in_process_all(start, direction)?;
            queue.push_back((start, direction));

            while let Some((nst, dir)) = queue.pop_front() {
                let cst = [nst, offset_position(nst, FLOOD_ST_MASK[dir], len)];
                let cs = [self.locate_leaf(cst[0])?, self.locate_leaf(cst[1])?];
                let sign = self.sign(cs[0], 0);
                for find in 0..4 {
                    let cind = FLOOD_FC_CELLS[find];
                    let Some(next_edge) =
                        self.neighbour_edge(cs[cind], FLOOD_FC_EDGES[dir][find], sign, false)
                    else {
                        continue;
                    };
                    let est = offset_position(cst[cind], VERTMAP[EDGEMAP[next_edge][0]], len);
                    let next_direction = next_edge / 4;
                    if (self.leaf(cs[cind]).flood_fill >> next_edge) & 1 == 0 {
                        self.set_in_process_all(est, next_direction)?;
                        queue.push_back((est, next_direction));
                        total += 1;
                    }
                }
            }

            if threshold == 0 {
                maximum = maximum.max(total);
                continue;
            }
            if total >= threshold {
                continue;
            }

            self.flip_parity_all(start, direction)?;
            queue.push_back((start, direction));
            while let Some((nst, dir)) = queue.pop_front() {
                let cst = [nst, offset_position(nst, FLOOD_ST_MASK[dir], len)];
                let cs = [self.locate_leaf(cst[0])?, self.locate_leaf(cst[1])?];
                let sign = self.sign(cs[0], 0);
                for find in 0..4 {
                    let cind = FLOOD_FC_CELLS[find];
                    let Some(next_edge) =
                        self.neighbour_edge(cs[cind], FLOOD_FC_EDGES[dir][find], sign, true)
                    else {
                        continue;
                    };
                    let est = offset_position(cst[cind], VERTMAP[EDGEMAP[next_edge][0]], len);
                    let next_direction = next_edge / 4;
                    if self.parity(cs[cind], next_edge) {
                        self.flip_parity_all(est, next_direction)?;
                        queue.push_back((est, next_direction));
                    }
                }
            }
        }
        Ok(maximum)
    }
}
