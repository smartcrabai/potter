// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   intern/dualcon/intern/octree.h and octree.cpp, SPDX-FileCopyrightText: 2011-2023 Blender Authors,
//   GPL-2.0-or-later.

//! Sparse octree storage, leaf helpers, triangle scan conversion, and edge-parity propagation.
//!
//! Storage model. Blender allocates `InternalNode`/`LeafNode` objects and rewrites a parent link
//! whenever a node is replaced (`addChild`, `addLeafChild`, `flipEdge`, `updateCell`, ...). Here
//! every node lives in an index arena and is mutated in place. A replacement in Blender only changes
//! the object identity; every later access in Blender goes through the parent link that was just
//! rewritten, so in-place mutation is observationally identical as long as no stale pointer is read,
//! and Blender's code never reads one. Children are stored by octant (`[Child; 8]`); Blender's
//! compacted child array is an equivalent view (`get_child(get_child_count(i))`).
//!
//! Nodes exist only where Blender creates them: `addTriangle` creates children only for
//! intersecting octants, and `locateCell` creates the missing cells that patching walks through.

use super::{
    invariant,
    projection::{CubeTriangleIsect, TriangleProjection},
    tables::{DIR_CELL, DIR_EDGE, VERTDIFF},
};
use crate::error::{ErrorCode, PotError, Result};

/// Combined input, octree, patch-path, and output item budget for one DualCon run.
pub(super) const MAX_DUALCON_WORK_ITEMS: usize = 4_000_000;

#[derive(Clone, Copy)]
pub(super) enum Arena {
    Internal,
    Leaf,
    Element,
    List,
    Vertex,
    Quad,
}

pub(super) fn allocation_limit() -> PotError {
    PotError::new(
        ErrorCode::LimitExceeded,
        "dual contour exceeded its combined allocation budget",
    )
}

pub(super) const GRID_DIMENSION: i32 = 20;

/// Blender `DualConMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    Centroid,
    MassPoint,
    Sharp,
}

/// Reference to a node slot: empty, a leaf arena index, or an internal arena index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Child {
    Empty,
    Leaf(u32),
    Internal(u32),
}

#[derive(Clone, Copy)]
pub(super) struct InternalNode {
    pub(super) children: [Child; 8],
}

/// `LeafNode`: edge parity bits, primary-edge mask, signs, minimizer index, and per-stored-edge
/// intersections (`EDGE_FLOATS` = offset, normal x, normal y, normal z).
#[derive(Clone, Copy)]
pub(super) struct LeafNode {
    /// Bit `e` is the parity of cube edge `e` (`edge_parity:12`).
    pub(super) edge_parity: u16,
    /// Bit `p` marks stored intersection on primary edge `p` (`primary_edge_intersections:3`).
    pub(super) primary: u8,
    /// Corner sign bits (`signs`).
    pub(super) signs: u8,
    pub(super) minimizer_index: i32,
    pub(super) flood_fill: u16,
    pub(super) edge_intersections: [[f32; 4]; 3],
}

impl LeafNode {
    pub(super) fn new() -> Self {
        Self {
            edge_parity: 0,
            primary: 0,
            signs: 0,
            minimizer_index: 0,
            flood_fill: 0,
            edge_intersections: [[0.0; 4]; 3],
        }
    }
}

/// `PathElement` as an arena node. Rings are circular through `next`.
#[derive(Clone, Copy)]
pub(super) struct PathElement {
    pub(super) pos: [i32; 3],
    pub(super) next: Option<u32>,
}

/// `PathList` as an arena node. Sentinel nodes (`head == None`) stand in for Blender's
/// `PathList *&` head variables so pointer-to-link updates map to plain field writes.
#[derive(Clone, Copy)]
pub(super) struct PathList {
    pub(super) head: Option<u32>,
    pub(super) tail: Option<u32>,
    pub(super) length: i32,
    pub(super) next: Option<u32>,
}

pub(super) struct Octree {
    pub(super) internals: Vec<InternalNode>,
    pub(super) leaves: Vec<LeafNode>,
    pub(super) elements: Vec<PathElement>,
    pub(super) lists: Vec<PathList>,
    pub(super) dimen: i32,
    pub(super) mindimen: i32,
    pub(super) minshift: i32,
    pub(super) max_depth: i32,
    pub(super) origin: [f32; 3],
    pub(super) range: f32,
    pub(super) thresh: f32,
    pub(super) hermite_num: f32,
    pub(super) mode: Mode,
    /// Output vertex positions in `add_vert` order.
    pub(super) vertices: Vec<[f32; 3]>,
    /// Output quads in `add_quad` order; entries are minimizer indices as Blender stores them.
    pub(super) quads: Vec<[i32; 4]>,
    work_items: usize,
}

impl Octree {
    /// Octree constructor (`Octree::Octree`). `origin` and `range` come from the input reader.
    pub(super) fn new(
        depth: i32,
        origin: [f32; 3],
        range: f32,
        threshold: f32,
        sharpness: f32,
        mode: Mode,
        input_work_items: usize,
    ) -> Result<Self> {
        let work_items = input_work_items
            .checked_add(1)
            .filter(|count| *count <= MAX_DUALCON_WORK_ITEMS)
            .ok_or_else(allocation_limit)?;
        let dimen = 1_i32 << GRID_DIMENSION;
        let mut internals = Vec::new();
        internals.try_reserve(1).map_err(|_| allocation_limit())?;
        internals.push(InternalNode {
            children: [Child::Empty; 8],
        });
        Ok(Self {
            internals,
            leaves: Vec::new(),
            elements: Vec::new(),
            lists: Vec::new(),
            dimen,
            mindimen: dimen >> depth,
            minshift: GRID_DIMENSION - depth,
            max_depth: depth,
            origin,
            range,
            thresh: threshold,
            hermite_num: sharpness,
            mode,
            vertices: Vec::new(),
            quads: Vec::new(),
            work_items,
        })
    }

    pub(super) fn check_work_items(&self, additional: usize) -> Result<()> {
        self.work_items
            .checked_add(additional)
            .filter(|count| *count <= MAX_DUALCON_WORK_ITEMS)
            .map(|_| ())
            .ok_or_else(allocation_limit)
    }

    pub(super) fn reserve_work_item(&mut self, arena: Arena) -> Result<()> {
        self.check_work_items(1)?;
        match arena {
            Arena::Internal => self.internals.try_reserve(1),
            Arena::Leaf => self.leaves.try_reserve(1),
            Arena::Element => self.elements.try_reserve(1),
            Arena::List => self.lists.try_reserve(1),
            Arena::Vertex => self.vertices.try_reserve(1),
            Arena::Quad => self.quads.try_reserve(1),
        }
        .map_err(|_| allocation_limit())?;
        self.work_items += 1;
        Ok(())
    }

    pub(super) fn push_vertex(&mut self, vertex: [f32; 3]) -> Result<()> {
        self.reserve_work_item(Arena::Vertex)?;
        self.vertices.push(vertex);
        Ok(())
    }

    pub(super) fn push_quad(&mut self, quad: [i32; 4]) -> Result<()> {
        self.reserve_work_item(Arena::Quad)?;
        self.quads.push(quad);
        Ok(())
    }

    pub(super) fn new_internal(&mut self) -> Result<u32> {
        self.reserve_work_item(Arena::Internal)?;
        self.internals.push(InternalNode {
            children: [Child::Empty; 8],
        });
        Ok((self.internals.len() - 1) as u32)
    }

    pub(super) fn new_leaf(&mut self) -> Result<u32> {
        self.reserve_work_item(Arena::Leaf)?;
        self.leaves.push(LeafNode::new());
        Ok((self.leaves.len() - 1) as u32)
    }

    pub(super) fn leaf(&self, id: u32) -> &LeafNode {
        &self.leaves[id as usize]
    }

    pub(super) fn leaf_mut(&mut self, id: u32) -> &mut LeafNode {
        &mut self.leaves[id as usize]
    }

    pub(super) fn root() -> u32 {
        0
    }

    /// `addAllTriangles`: scan-converts every input triangle in reader order, then runs the
    /// preparation passes that Blender runs before patching.
    pub(super) fn scan_convert(&mut self, triangles: &[[[f32; 3]; 3]]) -> Result<()> {
        for triangle in triangles {
            self.add_triangle(*triangle)?;
        }
        let root = Self::root();
        self.walk_cell(Child::Internal(root), super::walk::Pass::Parity)?;
        self.prepare_primal_edges_mask(root);
        Ok(())
    }

    /// `addTriangle(Triangle*, int)`: projects the triangle into grid space and inserts it.
    fn add_triangle(&mut self, vertices: [[f32; 3]; 3]) -> Result<()> {
        let dimen = self.dimen as f32;
        let mut grid = [[0_i64; 3]; 3];
        for (vertex, output) in vertices.iter().zip(grid.iter_mut()) {
            for axis in 0..3 {
                let projected = dimen * (vertex[axis] - self.origin[axis]) / self.range;
                output[axis] = projected as i64;
            }
        }
        let dimen64 = i64::from(self.dimen);
        let cube = [[0, 0, 0], [dimen64, dimen64, dimen64]];
        let (projection, inherit) = CubeTriangleIsect::root(cube, grid);
        let root = Self::root();
        self.add_triangle_node(root, &projection, &inherit, self.max_depth)
    }

    /// `addTriangle(InternalNode*, CubeTriangleIsect*, int)`.
    fn add_triangle_node(
        &mut self,
        node: u32,
        parent: &CubeTriangleIsect,
        inherit: &TriangleProjection,
        height: i32,
    ) -> Result<()> {
        let boxmask = parent.box_mask(inherit);
        let mut subp = parent.child();
        let mut tempdiff = [0_i32; 3];
        for (octant, diff) in VERTDIFF.iter().enumerate() {
            for axis in 0..3 {
                tempdiff[axis] += diff[axis];
            }
            if boxmask & (1 << octant) == 0 {
                continue;
            }
            subp.shift(tempdiff);
            tempdiff = [0; 3];

            if !subp.is_intersecting(inherit) {
                continue;
            }
            if self.internals[node as usize].children[octant] == Child::Empty {
                let child = if height == 1 {
                    Child::Leaf(self.new_leaf()?)
                } else {
                    Child::Internal(self.new_internal()?)
                };
                self.internals[node as usize].children[octant] = child;
            }
            match self.internals[node as usize].children[octant] {
                Child::Leaf(leaf) => self.update_cell(leaf, &subp, inherit),
                Child::Internal(child) => {
                    self.add_triangle_node(child, &subp, inherit, height - 1)?;
                }
                Child::Empty => return Err(invariant("dual contour child vanished during scan")),
            }
        }
        Ok(())
    }

    /// `updateCell`: records the primary-edge crossings of one triangle in a leaf.
    fn update_cell(
        &mut self,
        leaf: u32,
        triangle: &CubeTriangleIsect,
        inherit: &TriangleProjection,
    ) {
        const PRIMARY: [usize; 3] = [0, 4, 8];
        let mut old_count = 0_usize;
        let mut new_count = 0_usize;
        let mut offsets = [0.0_f32; 3];
        let mut normals = [[0.0_f32; 3]; 3];
        let normal = inherit.norm.map(|value| value as f32);
        for (axis, &edge) in PRIMARY.iter().enumerate() {
            if self.parity(leaf, edge) {
                let stored = self.leaf(leaf).edge_intersections[old_count];
                offsets[new_count] = stored[0];
                normals[new_count] = [stored[1], stored[2], stored[3]];
                old_count += 1;
                new_count += 1;
            } else if triangle.is_intersecting_primary(axis, inherit) {
                self.set_edge(leaf, edge);
                offsets[new_count] = triangle.intersection_primary(axis, inherit);
                normals[new_count] = normal;
                new_count += 1;
            }
        }
        if new_count > old_count {
            let node = self.leaf_mut(leaf);
            for index in 0..new_count {
                node.edge_intersections[index] = [
                    offsets[index],
                    normals[index][0],
                    normals[index][1],
                    normals[index][2],
                ];
            }
        }
    }

    /// `preparePrimalEdgesMask`: derives the primary-edge mask from edge parity.
    fn prepare_primal_edges_mask(&mut self, node: u32) {
        for octant in 0..8 {
            match self.internals[node as usize].children[octant] {
                Child::Empty => {}
                Child::Leaf(leaf) => {
                    let parity = self.leaf(leaf).edge_parity;
                    let primary = (parity & 0x1) | ((parity & 0x10) >> 3) | ((parity & 0x100) >> 6);
                    self.leaf_mut(leaf).primary = primary as u8;
                }
                Child::Internal(child) => self.prepare_primal_edges_mask(child),
            }
        }
    }

    // ----- leaf edge helpers ---------------------------------------------------------------

    pub(super) fn parity(&self, leaf: u32, index: usize) -> bool {
        (self.leaf(leaf).edge_parity >> index) & 1 == 1
    }

    pub(super) fn set_edge(&mut self, leaf: u32, index: usize) {
        self.leaf_mut(leaf).edge_parity |= 1 << index;
    }

    /// `flipEdge(LeafNode*, int)`: toggles parity only.
    pub(super) fn flip_edge(&mut self, leaf: u32, index: usize) {
        self.leaf_mut(leaf).edge_parity ^= 1 << index;
    }

    pub(super) fn stored(&self, leaf: u32, primary_index: usize) -> bool {
        (self.leaf(leaf).primary >> primary_index) & 1 == 1
    }

    pub(super) fn set_stored(&mut self, leaf: u32, primary_index: usize) {
        self.leaf_mut(leaf).primary |= 1 << primary_index;
    }

    /// `getEdgeCount`: index into `edge_intersections` for a stored primary edge.
    pub(super) fn edge_count(&self, leaf: u32, primary_index: usize) -> usize {
        let mask = self.leaf(leaf).primary as usize;
        (mask & ((1 << primary_index) - 1)).count_ones() as usize
    }

    /// `flipEdge(LeafNode*, int, float)`: toggles parity and, for a primary edge that becomes set
    /// without a stored intersection, inserts `alpha` at its sorted position.
    pub(super) fn flip_edge_with_offset(&mut self, leaf: u32, index: usize, alpha: f32) {
        self.flip_edge(leaf, index);
        if index.is_multiple_of(4) {
            let primary_index = index / 4;
            if self.parity(leaf, index) && !self.stored(leaf, primary_index) {
                let stored_count = self.leaf(leaf).primary.count_ones() as usize;
                self.set_stored(leaf, primary_index);
                let count = self.edge_count(leaf, primary_index);
                let node = self.leaf_mut(leaf);
                let mut position = stored_count;
                while position > count {
                    node.edge_intersections[position] = node.edge_intersections[position - 1];
                    position -= 1;
                }
                node.edge_intersections[count] = [alpha, 0.0, 0.0, 0.0];
            }
        }
    }

    /// `getEdgeOffset`.
    pub(super) fn edge_offset(&self, leaf: u32, count: usize) -> f32 {
        self.leaf(leaf).edge_intersections[count][0]
    }

    /// `getEdgeIntersectionByIndex(leaf, index, st, len, pt, nm)`.
    pub(super) fn edge_intersection(
        &self,
        leaf: u32,
        index: usize,
        st: [i32; 3],
        len: i32,
    ) -> ([f32; 3], [f32; 3]) {
        let count = self.edge_count(leaf, index);
        let stored = self.leaf(leaf).edge_intersections[count];
        let mut point = st.map(|coordinate| coordinate as f32);
        point[index] += stored[0] * len as f32;
        (point, [stored[1], stored[2], stored[3]])
    }

    /// `getEdgeIntersectionByIndex(st, index, pt, check=1)` used by `getFacePoint`.
    pub(super) fn edge_intersection_checked(&self, st: [i32; 3], index: usize) -> Option<[f32; 3]> {
        let leaf = self.locate_leaf_check(st)?;
        if !self.stored(leaf, index) {
            return None;
        }
        let offset = self.edge_offset(leaf, self.edge_count(leaf, index));
        let mut point = st.map(|coordinate| coordinate as f32);
        point[index] += offset * self.mindimen as f32;
        Some(point)
    }

    /// Corner sign bit (`getSign(const LeafNode*, index)`).
    pub(super) fn sign(&self, leaf: u32, index: usize) -> u8 {
        (self.leaf(leaf).signs >> index) & 1
    }

    /// `getFaceParity`: parity of a face's four edges.
    pub(super) fn face_parity(&self, leaf: u32, face: usize) -> bool {
        let sum: usize = (0..4)
            .map(|edge| usize::from(self.parity(leaf, super::tables::FACE_MAP[face][edge])))
            .sum();
        sum & 1 == 1
    }

    // ----- navigation ----------------------------------------------------------------------

    /// `locateLeaf(st)`: leaf at a grid position, expected to exist.
    pub(super) fn locate_leaf(&self, st: [i32; 3]) -> Result<u32> {
        self.locate_leaf_check(st)
            .ok_or_else(|| invariant("dual contour leaf lookup missed an expected cell"))
    }

    /// `locateLeafCheck(st)`.
    pub(super) fn locate_leaf_check(&self, st: [i32; 3]) -> Option<u32> {
        let mut node = Self::root();
        for bit in (GRID_DIMENSION - self.max_depth..GRID_DIMENSION).rev() {
            let index = octant_index(st, bit);
            match self.internals[node as usize].children[index] {
                Child::Internal(child) => node = child,
                Child::Leaf(leaf) => return Some(leaf),
                Child::Empty => return None,
            }
        }
        None
    }

    /// `locateLeaf(InternalNode *parent, int len, st)`: descends from `parent` (size `len`).
    pub(super) fn locate_leaf_below(&self, parent: u32, len: i32, st: [i32; 3]) -> Result<u32> {
        let mut node = Child::Internal(parent);
        let mut shift = len / 2;
        while shift >= self.mindimen {
            let index = octant_bit(st, shift);
            node = match node {
                Child::Internal(id) => self.internals[id as usize].children[index],
                _ => {
                    return Err(invariant(
                        "dual contour leaf walk expected an internal node",
                    ));
                }
            };
            shift >>= 1;
        }
        match node {
            Child::Leaf(leaf) => Ok(leaf),
            _ => Err(invariant("dual contour leaf walk ended without a leaf")),
        }
    }

    /// `setInProcessAll`: marks the four leaves around an edge as visited by flood fill.
    pub(super) fn set_in_process_all(&mut self, st: [i32; 3], direction: usize) -> Result<()> {
        for cell in 0..4 {
            let neighbour = offset_position(st, DIR_CELL[direction][cell], self.mindimen);
            let leaf = self.locate_leaf_check(neighbour).ok_or_else(|| {
                invariant("dual contour flood fill reached a missing cell around an edge")
            })?;
            self.leaf_mut(leaf).flood_fill |= 1 << DIR_EDGE[direction][cell];
        }
        Ok(())
    }

    /// `flipParityAll`: toggles parity of the four leaves around an edge.
    pub(super) fn flip_parity_all(&mut self, st: [i32; 3], direction: usize) -> Result<()> {
        for cell in 0..4 {
            let neighbour = offset_position(st, DIR_CELL[direction][cell], self.mindimen);
            let leaf = self.locate_leaf(neighbour)?;
            self.flip_edge(leaf, DIR_EDGE[direction][cell]);
        }
        Ok(())
    }

    /// `processEdgeParity`: if any copy of a minimal edge is set, set it on all four leaves.
    pub(super) fn process_edge_parity(&mut self, leaves: [u32; 4], direction: usize) {
        let masks = super::tables::PROCESS_EDGE_MASK[direction];
        let any = (0..4).any(|index| self.parity(leaves[index], masks[index]));
        if any {
            for index in 0..4 {
                self.set_edge(leaves[index], masks[index]);
            }
        }
    }
}

/// Octant index of `st` at bit `bit` (`locateLeafCheck` / `locateLeaf` bit order).
pub(super) fn octant_index(st: [i32; 3], bit: i32) -> usize {
    (((st[0] >> bit) & 1) << 2 | ((st[1] >> bit) & 1) << 1 | ((st[2] >> bit) & 1)) as usize
}

/// Octant index of `st` at mask value `mask` (`locateLeaf(parent, len, st)` bit order).
pub(super) fn octant_bit(st: [i32; 3], mask: i32) -> usize {
    usize::from(st[0] & mask != 0) * 4
        + usize::from(st[1] & mask != 0) * 2
        + usize::from(st[2] & mask != 0)
}

pub(super) fn offset_position(st: [i32; 3], offset: [i32; 3], size: i32) -> [i32; 3] {
    std::array::from_fn(|axis| st[axis] + offset[axis] * size)
}

/// Children of an internal node slot, or all-empty for leaves and empty slots.
pub(super) fn child_slots(internals: &[InternalNode], node: Child) -> [Child; 8] {
    match node {
        Child::Internal(id) => internals[id as usize].children,
        _ => [Child::Empty; 8],
    }
}
