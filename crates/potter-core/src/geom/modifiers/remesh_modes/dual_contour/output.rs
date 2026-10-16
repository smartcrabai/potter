// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   intern/dualcon/intern/octree.cpp (writeOut, generateMinimizer, computeMinimizer,
//   fillEdgeIntersections, processEdgeWrite), SPDX-FileCopyrightText: 2011-2023 Blender Authors,
//   GPL-2.0-or-later.

//! Output generation: one vertex per mixed cell (in tree order) and one quad per intersecting
//! primary edge, emitted in Blender's `dualcon_add_vert` / `dualcon_add_quad` order.

use super::{
    octree::{Child, Mode, Octree, offset_position},
    qef::{mass_point, minimize},
    tables::{EDGEMAP, PROCESS_EDGE_MASK, VERTMAP},
    walk::Pass,
};
use crate::error::Result;

/// Primary-edge slot of each cube edge touched by the minimal-cell neighbourhood.
const PRIMARY_SLOT: [usize; 3] = [0, 4, 8];
/// Face neighbour edges: `(neighbour edge, neighbour primary index)` per axis.
const FACE_NEIGHBOUR_EDGES: [[usize; 2]; 3] = [[6, 10], [2, 9], [1, 5]];
const FACE_NEIGHBOUR_PRIMARY: [[usize; 2]; 3] = [[1, 2], [0, 2], [0, 1]];
/// Edge-adjacent neighbour edges: the cell that shares the edge along each axis.
const EDGE_NEIGHBOUR_EDGES: [usize; 3] = [3, 7, 11];
const EDGE_NEIGHBOUR_PRIMARY: [usize; 3] = [0, 1, 2];

type Crossings = ([[f32; 3]; 12], [[f32; 3]; 12], [bool; 12]);

impl Octree {
    /// `writeOut()`: builds the minimizer vertices and the quads from the signed grid.
    pub(super) fn write_out(&mut self) -> Result<()> {
        let root = Child::Internal(Self::root());
        let mut offset = 0_i32;
        self.generate_minimizer(root, [0; 3], self.dimen, &mut offset)?;
        self.walk_cell(root, Pass::Contour)?;
        Ok(())
    }

    /// `generateMinimizer`: emits one vertex for each leaf whose sign mask is mixed.
    fn generate_minimizer(
        &mut self,
        node: Child,
        st: [i32; 3],
        len: i32,
        offset: &mut i32,
    ) -> Result<()> {
        match node {
            Child::Empty => {}
            Child::Leaf(leaf) => {
                let center = st.map(|coordinate| coordinate as f32 + (len / 2) as f32);
                let mut rvalue = self.compute_minimizer(leaf, st, len, center);
                for (axis, coordinate) in rvalue.iter_mut().enumerate() {
                    *coordinate = *coordinate * self.range / self.dimen as f32 + self.origin[axis];
                }
                let sign_mask = self.leaf(leaf).signs;
                let multiplicity = i32::from(sign_mask > 0 && sign_mask < 255);
                for _ in 0..multiplicity {
                    self.push_vertex(rvalue)?;
                }
                self.leaf_mut(leaf).minimizer_index = *offset;
                *offset += multiplicity;
            }
            Child::Internal(id) => {
                let children = self.internals[id as usize].children;
                let child_len = len >> 1;
                for (octant, child) in children.iter().enumerate() {
                    if *child == Child::Empty {
                        continue;
                    }
                    let child_origin = offset_position(st, VERTMAP[octant], child_len);
                    self.generate_minimizer(*child, child_origin, child_len, offset)?;
                }
            }
        }
        Ok(())
    }

    /// `computeMinimizer`: the vertex position of one cell in grid units.
    fn compute_minimizer(&self, leaf: u32, st: [i32; 3], len: i32, center: [f32; 3]) -> [f32; 3] {
        let (points, normals, parity) = self.fill_edge_intersections(leaf, st, len);
        match self.mode {
            Mode::Centroid => center,
            Mode::MassPoint => {
                let mut mass = [0.0_f32; 3];
                mass_point(&mut mass, &points, &parity);
                mass
            }
            Mode::Sharp => {
                let mut rvalue = center;
                let mut mass = [0.0_f32; 3];
                minimize(&mut rvalue, &mut mass, &points, &normals, &parity);

                // Restrain the minimizer to the cell and its hermite margin.
                let near = self.hermite_num * len as f32;
                let far = (1.0 + self.hermite_num) * len as f32;
                let outside = (0..3).any(|axis| {
                    let origin = st[axis] as f32;
                    rvalue[axis] < origin - near || rvalue[axis] > origin + far
                });
                if outside { mass } else { rvalue }
            }
        }
    }

    /// `fillEdgeIntersections`: gathers the crossings on the twelve edges of a cell, reading the
    /// shared edges from the neighbouring leaves (positions are relative to each owning leaf).
    fn fill_edge_intersections(&self, leaf: u32, st: [i32; 3], len: i32) -> Crossings {
        let mut points = [[0.0_f32; 3]; 12];
        let mut normals = [[0.0_f32; 3]; 12];
        let mut parity = [false; 12];

        let mut record = |slot: usize, point: [f32; 3], normal: [f32; 3]| {
            points[slot] = point;
            normals[slot] = normal;
            parity[slot] = true;
        };

        for (axis, &slot) in PRIMARY_SLOT.iter().enumerate() {
            if self.stored(leaf, axis) {
                let (point, normal) = self.edge_intersection(leaf, axis, st, len);
                record(slot, point, normal);
            }
        }

        for axis in 0..3 {
            let mut neighbour_origin = st;
            neighbour_origin[axis] += len;
            let Some(neighbour) = self.locate_leaf_check(neighbour_origin) else {
                continue;
            };
            for side in 0..2 {
                let primary = FACE_NEIGHBOUR_PRIMARY[axis][side];
                if self.stored(neighbour, primary) {
                    let (point, normal) =
                        self.edge_intersection(neighbour, primary, neighbour_origin, len);
                    record(FACE_NEIGHBOUR_EDGES[axis][side], point, normal);
                }
            }
        }

        for axis in 0..3 {
            let mut neighbour_origin = st.map(|coordinate| coordinate + len);
            neighbour_origin[axis] -= len;
            let Some(neighbour) = self.locate_leaf_check(neighbour_origin) else {
                continue;
            };
            let primary = EDGE_NEIGHBOUR_PRIMARY[axis];
            if self.stored(neighbour, primary) {
                let (point, normal) =
                    self.edge_intersection(neighbour, primary, neighbour_origin, len);
                record(EDGE_NEIGHBOUR_EDGES[axis], point, normal);
            }
        }

        (points, normals, parity)
    }

    /// `processEdgeWrite`: emits a quad for an intersecting primary edge shared by four leaves.
    pub(super) fn process_edge_write(&mut self, leaves: [u32; 4], direction: usize) -> Result<()> {
        let masks = PROCESS_EDGE_MASK[direction];
        if !self.parity(leaves[3], masks[3]) {
            return Ok(());
        }
        let edge = masks[3];
        let flip = self.sign(leaves[3], EDGEMAP[edge][1]) > 0;
        let index = |leaf: usize| self.leaf(leaves[leaf]).minimizer_index;
        let quad = if flip {
            [index(2), index(3), index(1), index(0)]
        } else {
            [index(0), index(1), index(3), index(2)]
        };
        self.push_quad(quad)
    }
}
