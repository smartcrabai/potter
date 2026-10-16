// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   intern/dualcon/intern/octree.h and octree.cpp, SPDX-FileCopyrightText: 2011-2023 Blender Authors,
//   GPL-2.0-or-later. Constant tables are copied verbatim (indices are cell/edge/corner numbers).

//! Lookup tables of Blender's dual-contouring octree (`Projections.cpp` and `octree.cpp`).

/// Child corner offsets in units of the child size: `vertmap[child][axis]`.
pub(super) const VERTMAP: [[i32; 3]; 8] = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
];

/// Cube edge endpoints: `edgemap[edge] = [corner_a, corner_b]`.
pub(super) const EDGEMAP: [[usize; 2]; 12] = [
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
];

pub(super) const FACE_MAP: [[usize; 4]; 6] = [
    [4, 8, 5, 9],
    [6, 10, 7, 11],
    [0, 8, 1, 10],
    [2, 9, 3, 11],
    [0, 4, 2, 6],
    [1, 5, 3, 7],
];

pub(super) const CELL_PROC_FACE_MASK: [[usize; 3]; 12] = [
    [0, 4, 0],
    [1, 5, 0],
    [2, 6, 0],
    [3, 7, 0],
    [0, 2, 1],
    [4, 6, 1],
    [1, 3, 1],
    [5, 7, 1],
    [0, 1, 2],
    [2, 3, 2],
    [4, 5, 2],
    [6, 7, 2],
];

pub(super) const CELL_PROC_EDGE_MASK: [[usize; 5]; 6] = [
    [0, 1, 2, 3, 0],
    [4, 5, 6, 7, 0],
    [0, 4, 1, 5, 1],
    [2, 6, 3, 7, 1],
    [0, 2, 4, 6, 2],
    [1, 3, 5, 7, 2],
];

pub(super) const FACE_PROC_FACE_MASK: [[[usize; 3]; 4]; 3] = [
    [[4, 0, 0], [5, 1, 0], [6, 2, 0], [7, 3, 0]],
    [[2, 0, 1], [6, 4, 1], [3, 1, 1], [7, 5, 1]],
    [[1, 0, 2], [3, 2, 2], [5, 4, 2], [7, 6, 2]],
];

pub(super) const FACE_PROC_EDGE_MASK: [[[usize; 6]; 4]; 3] = [
    [
        [1, 4, 0, 5, 1, 1],
        [1, 6, 2, 7, 3, 1],
        [0, 4, 6, 0, 2, 2],
        [0, 5, 7, 1, 3, 2],
    ],
    [
        [0, 2, 3, 0, 1, 0],
        [0, 6, 7, 4, 5, 0],
        [1, 2, 0, 6, 4, 2],
        [1, 3, 1, 7, 5, 2],
    ],
    [
        [1, 1, 0, 3, 2, 0],
        [1, 5, 4, 7, 6, 0],
        [0, 1, 5, 0, 4, 1],
        [0, 3, 7, 2, 6, 1],
    ],
];

pub(super) const EDGE_PROC_EDGE_MASK: [[[usize; 5]; 2]; 3] = [
    [[3, 2, 1, 0, 0], [7, 6, 5, 4, 0]],
    [[5, 1, 4, 0, 1], [7, 3, 6, 2, 1]],
    [[6, 4, 2, 0, 2], [7, 5, 3, 1, 2]],
];

pub(super) const PROCESS_EDGE_MASK: [[usize; 4]; 3] = [[3, 2, 1, 0], [7, 5, 6, 4], [11, 10, 9, 8]];

pub(super) const DIR_CELL: [[[i32; 3]; 4]; 3] = [
    [[0, -1, -1], [0, -1, 0], [0, 0, -1], [0, 0, 0]],
    [[-1, 0, -1], [-1, 0, 0], [0, 0, -1], [0, 0, 0]],
    [[-1, -1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 0]],
];

pub(super) const DIR_EDGE: [[usize; 4]; 3] = [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]];

/// Child index offsets of `trace`/`patch` split planes (`vertdiff` in `addTriangle`).
pub(super) const VERTDIFF: [[i32; 3]; 8] = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 0, 1],
    [1, -1, -1],
    [0, 0, 1],
    [0, 1, -1],
    [0, 0, 1],
];

/// Edge-parity neighbourhood tables used by `flood_fill` (`fcEdges` in `floodFill`).
pub(super) const FLOOD_FC_CELLS: [usize; 4] = [1, 0, 1, 0];
pub(super) const FLOOD_FC_EDGES: [[[usize; 3]; 4]; 3] = [
    [[9, 2, 11], [8, 1, 10], [5, 1, 7], [4, 2, 6]],
    [[10, 6, 11], [8, 5, 9], [1, 5, 3], [0, 6, 2]],
    [[6, 10, 7], [4, 9, 5], [2, 9, 3], [0, 10, 1]],
];

/// Neighbour-leaf offsets for `flood_fill` (`stMask` in `floodFill`), scaled by the leaf size.
pub(super) const FLOOD_ST_MASK: [[i32; 3]; 3] = [[0, -1, -1], [-1, 0, -1], [-1, -1, 0]];
