// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from OpenVDB (https://github.com/AcademySoftwareFoundation/openvdb):
//   openvdb/openvdb/tools/VolumeToMesh.h (v13.0.0), SPDX-FileCopyrightText: Copyright Contributors to the OpenVDB Project, Apache-2.0.
//   openvdb/openvdb/math/Operators.h, openvdb/openvdb/math/Vec3.h (v13.0.0).
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/mesh_remesh_voxel.cc, SPDX-FileCopyrightText: 2019 Blender Authors, GPL-2.0-or-later.

//! Adaptive `volumeToMesh` for Blender's VOXEL remesh (adaptivity above zero).
//!
//! This ports `OpenVDB` 13.0.0 `VolumeToMesh` for the configuration Blender uses:
//! adaptive meshing with no reference grid, no surface or spatial masks, and
//! `relaxDisorientedTriangles = false`. The stages run in `OpenVDB` order:
//!
//! 1. Cell sign flags with ambiguous-face correction (`computeAuxiliaryData`). The INSIDE bit
//!    comes from the uncorrected corner 0, while the sign bits use the corrected configuration.
//! 2. Region merging over 2^3, 4^3 and 8^3 blocks inside each 8^3 leaf (`MergeVoxelRegions`),
//!    gated by non-manifold tests and, below adaptivity 1, by gradient agreement.
//! 3. Per-leaf point generation (`ComputePoints`): one point per unmerged cell edge group, then
//!    one averaged point per merged region in ascending region id.
//! 4. Quads and triangles per cell edge (`ComputePolygons` with `AdaptivePrimBuilder`).
//!
//! Leaves are visited in `OpenVDB` tree order so vertex and polygon order follow `OpenVDB`.
//! Each cell is classified from its corner samples; `OpenVDB` additionally skips cells whose
//! sign-changing edges have no active endpoint, which cannot occur for a narrow-band level set.

use std::collections::{BTreeMap, HashMap};

use glam::DVec3;

use super::super::voxel_edge_groups::{AMBIGUOUS_FACE, EDGE_GROUPS, SADAPTABLE};

const INSIDE: u16 = 0x100;
const XEDGE: u16 = 0x200;
const YEDGE: u16 = 0x400;
const ZEDGE: u16 = 0x800;
const EDGES: u16 = XEDGE | YEDGE | ZEDGE;
const LEAF_DIM: i64 = 8;
const LEAF_CELLS: usize = 512;

/// Corner offsets of a cell in `OpenVDB`'s order.
const CORNERS: [[i64; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 1],
];

/// Adaptive surface in `OpenVDB`'s convention; Blender's corner remap is applied by the caller.
pub(super) struct Surface {
    pub(super) positions: Vec<DVec3>,
    pub(super) quads: Vec<[usize; 4]>,
    pub(super) triangles: Vec<[usize; 3]>,
}

/// Dense level-set samples addressed by absolute index-space coordinates.
struct Field<'a> {
    values: &'a [f32],
    dimensions: [usize; 3],
    min_coord: [i64; 3],
    background: f32,
}

impl Field<'_> {
    fn value(&self, coord: [i64; 3]) -> f32 {
        match dense_index(coord, self.min_coord, self.dimensions) {
            Some(index) => self.values[index],
            None => self.background,
        }
    }
}

/// Cell flags over the cell lattice (`points - 1`), indexed like the point grid.
struct Cells<'a> {
    field: Field<'a>,
    dimensions: [usize; 3],
    flags: Vec<u16>,
}

impl Cells<'_> {
    fn coord(&self, index: usize) -> [i64; 3] {
        let plane = self.dimensions[1] * self.dimensions[2];
        let local = [
            index / plane,
            (index / self.dimensions[2]) % self.dimensions[1],
            index % self.dimensions[2],
        ];
        add(self.field.min_coord, local.map(|c| c as i64))
    }

    fn flag(&self, coord: [i64; 3]) -> u16 {
        dense_index(coord, self.field.min_coord, self.dimensions)
            .map_or(0, |index| self.flags[index])
    }

    fn is_on(&self, coord: [i64; 3]) -> bool {
        self.flag(coord) != 0
    }
}

struct Leaf {
    origin: [i64; 3],
    /// Cell indices of this leaf's ON cells, in leaf-local offset order.
    cells: Vec<usize>,
}

fn add(a: [i64; 3], b: [i64; 3]) -> [i64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [i64; 3], b: [i64; 3]) -> [i64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dense_index(coord: [i64; 3], min_coord: [i64; 3], dimensions: [usize; 3]) -> Option<usize> {
    let mut local = [0_usize; 3];
    for axis in 0..3 {
        let offset = usize::try_from(coord[axis] - min_coord[axis]).ok()?;
        if offset >= dimensions[axis] {
            return None;
        }
        local[axis] = offset;
    }
    Some((local[0] * dimensions[1] + local[1]) * dimensions[2] + local[2])
}

/// Offset of a leaf-local coordinate (each component in `0..8`) in `OpenVDB`'s leaf order.
fn local_offset(local: [i64; 3]) -> usize {
    ((local[0] << 6) | (local[1] << 3) | local[2]) as usize
}

fn corner_signs(field: &Field, origin: [i64; 3]) -> u8 {
    let mut signs = 0_u8;
    for (bit, offset) in CORNERS.iter().enumerate() {
        if field.value(add(origin, *offset)) < 0.0 {
            signs |= 1 << bit;
        }
    }
    signs
}

/// `correctCellSigns`: complements the cell when the neighbor across its ambiguous face agrees.
fn correct_signs(field: &Field, signs: u8, face: u8, origin: [i64; 3]) -> u8 {
    let (offset, expected) = match face {
        1 => ([0, 0, -1], 3),
        2 => ([1, 0, 0], 4),
        3 => ([0, 0, 1], 1),
        4 => ([-1, 0, 0], 2),
        5 => ([0, -1, 0], 6),
        6 => ([0, 1, 0], 5),
        _ => return signs,
    };
    let neighbor = corner_signs(field, add(origin, offset));
    if AMBIGUOUS_FACE[usize::from(neighbor)] == expected {
        !signs
    } else {
        signs
    }
}

/// `computeAuxiliaryData` for one cell: returns 0 for cells that do not intersect the surface.
fn cell_flags(field: &Field, origin: [i64; 3]) -> u16 {
    let raw = corner_signs(field, origin);
    if raw == 0 || raw == 0xFF {
        return 0;
    }
    let inside = (raw & 1) != 0;
    let mut flags = if inside { INSIDE } else { 0 };
    if inside != ((raw & 0x02) != 0) {
        flags |= XEDGE;
    }
    if inside != ((raw & 0x10) != 0) {
        flags |= YEDGE;
    }
    if inside != ((raw & 0x08) != 0) {
        flags |= ZEDGE;
    }
    let face = AMBIGUOUS_FACE[usize::from(raw)];
    let signs = if face == 0 {
        raw
    } else {
        correct_signs(field, raw, face, origin)
    };
    flags | u16::from(signs)
}

/// `isNonManifold`: true when a `dim`-sized block at `origin` may contain a topological ambiguity.
fn is_nonmanifold(field: &Field, origin: [i64; 3], dim: i64) -> bool {
    let inside = |coord: [i64; 3]| field.value(coord) < 0.0;
    let half = dim >> 1;
    let [i, j, k] = origin;
    let (ip, ipp) = (i + half, i + dim);
    let (jp, jpp) = (j + half, j + dim);
    let (kp, kpp) = (k + half, k + dim);
    let p = [
        inside([i, j, k]),
        inside([ipp, j, k]),
        inside([ipp, j, kpp]),
        inside([i, j, kpp]),
        inside([i, jpp, k]),
        inside([ipp, jpp, k]),
        inside([ipp, jpp, kpp]),
        inside([i, jpp, kpp]),
    ];
    let mut signs = 0_u8;
    for (bit, value) in p.iter().enumerate() {
        if *value {
            signs |= 1 << bit;
        }
    }
    if !SADAPTABLE[usize::from(signs)] {
        return true;
    }
    let edges: [([i64; 3], usize, usize); 12] = [
        ([ip, j, k], 0, 1),
        ([ipp, j, kp], 1, 2),
        ([ip, j, kpp], 2, 3),
        ([i, j, kp], 0, 3),
        ([ip, jpp, k], 4, 5),
        ([ipp, jpp, kp], 5, 6),
        ([ip, jpp, kpp], 6, 7),
        ([i, jpp, kp], 7, 4),
        ([i, jp, k], 0, 4),
        ([ipp, jp, k], 1, 5),
        ([ipp, jp, kpp], 2, 6),
        ([i, jp, kpp], 3, 7),
    ];
    if edges
        .iter()
        .any(|&(coord, a, b)| p[a] != inside(coord) && p[b] != inside(coord))
    {
        return true;
    }
    let faces: [([i64; 3], [usize; 4]); 6] = [
        ([ip, jp, k], [0, 1, 4, 5]),
        ([ipp, jp, kp], [1, 2, 5, 6]),
        ([ip, jp, kpp], [2, 3, 6, 7]),
        ([i, jp, kp], [0, 3, 4, 7]),
        ([ip, j, kp], [0, 1, 2, 3]),
        ([ip, jpp, kp], [4, 5, 6, 7]),
    ];
    if faces.iter().any(|(coord, corners)| {
        let m = inside(*coord);
        corners.iter().all(|&corner| p[corner] != m)
    }) {
        return true;
    }
    let m = inside([ip, jp, kp]);
    p.iter().all(|&corner| corner != m)
}

/// `ISGradient<CD_2ND>` followed by `Vec3<float>::normalize`.
fn normalized_gradient(field: &Field, coord: [i64; 3]) -> [f32; 3] {
    let half = 0.5_f32;
    let mut dir = [
        (field.value(add(coord, [1, 0, 0])) - field.value(add(coord, [-1, 0, 0]))) * half,
        (field.value(add(coord, [0, 1, 0])) - field.value(add(coord, [0, -1, 0]))) * half,
        (field.value(add(coord, [0, 0, 1])) - field.value(add(coord, [0, 0, -1]))) * half,
    ];
    let length = ((dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) as f64).sqrt() as f32;
    if length > 1.0e-7_f32 {
        let inverse = 1.0_f32 / length;
        dir = dir.map(|component| component * inverse);
    }
    dir
}

/// `isMergeable`: every pair of gradients in the block agrees within `adaptivity`.
fn is_mergeable(
    gradient: &[[f32; 3]; LEAF_CELLS],
    has_gradient: &[bool; LEAF_CELLS],
    local: [i64; 3],
    dim: i64,
    adaptivity: f32,
) -> bool {
    if f64::from(adaptivity) < 1.0e-6 {
        return false;
    }
    let mut norm_slots = [0_usize; LEAF_CELLS];
    let mut norm_count = 0;
    for dx in 0..dim {
        for dy in 0..dim {
            for dz in 0..dim {
                let slot = local_offset([local[0] + dx, local[1] + dy, local[2] + dz]);
                if has_gradient[slot] {
                    norm_slots[norm_count] = slot;
                    norm_count += 1;
                }
            }
        }
    }
    for &a_slot in &norm_slots[..norm_count] {
        let a = gradient[a_slot];
        for &b_slot in &norm_slots[..norm_count] {
            let b = gradient[b_slot];
            let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            if 1.0 - f64::from(dot) > f64::from(adaptivity) {
                return false;
            }
        }
    }
    true
}

/// `MergeVoxelRegions` for one leaf. Returns the region id of each leaf-local cell (0 = unmerged).
fn merge_leaf(
    cells: &Cells,
    leaf: &Leaf,
    adaptivity: f32,
    use_gradients: bool,
) -> [u32; LEAF_CELLS] {
    let origin = leaf.origin;
    let mut mask = [false; LEAF_CELLS];

    // Sign-based masks: non-adaptable or multi-group cells, and cells whose shared edges lack
    // neighbors on the surface.
    for &index in &leaf.cells {
        let coord = cells.coord(index);
        let local = sub(coord, origin);
        let flag = cells.flags[index];
        let signs = (flag & 0xFF) as u8;
        let block = local.map(|c| c & !1);
        if !SADAPTABLE[usize::from(signs)] || EDGE_GROUPS[usize::from(signs)][0] > 1 {
            mask[local_offset(block)] = true;
        } else if (flag & EDGES) != 0 {
            let on = |offset: [i64; 3]| cells.is_on(add(coord, offset));
            let mut masked = false;
            if (flag & XEDGE) != 0 {
                masked |= !(on([0, -1, 0]) && on([0, -1, -1]) && on([0, 0, -1]));
            }
            if (flag & YEDGE) != 0 {
                masked |= !(on([0, 0, -1]) && on([-1, 0, -1]) && on([-1, 0, 0]));
            }
            if (flag & ZEDGE) != 0 {
                masked |= !(on([0, -1, 0]) && on([-1, -1, 0]) && on([-1, 0, 0]));
            }
            if masked {
                mask[local_offset(block)] = true;
            }
        }
    }

    // Topologically ambiguous 2x2x2 sub-blocks.
    for lx in (0..LEAF_DIM).step_by(2) {
        for ly in (0..LEAF_DIM).step_by(2) {
            for lz in (0..LEAF_DIM).step_by(2) {
                let local = [lx, ly, lz];
                let slot = local_offset(local);
                if !mask[slot] && is_nonmanifold(&cells.field, add(origin, local), 2) {
                    mask[slot] = true;
                }
            }
        }
    }

    // Gradients for ON cells whose 2x2x2 block is not masked.
    let mut gradient = [[0.0_f32; 3]; LEAF_CELLS];
    let mut has_gradient = [false; LEAF_CELLS];
    if use_gradients {
        for &index in &leaf.cells {
            let coord = cells.coord(index);
            let local = sub(coord, origin);
            if !mask[local_offset(local.map(|c| c & !1))] {
                let slot = local_offset(local);
                gradient[slot] = normalized_gradient(&cells.field, coord);
                has_gradient[slot] = true;
            }
        }
    }

    // Merge blocks of 2, 4, then 8 (the leaf) wherever possible; a blocked block masks its parent.
    let mut regions = [0_u32; LEAF_CELLS];
    let mut region_id = 1_u32;
    let mut dim = 2_i64;
    while dim <= LEAF_DIM {
        let coord_mask = !((dim << 1) - 1);
        for lx in (0..LEAF_DIM).step_by(dim as usize) {
            for ly in (0..LEAF_DIM).step_by(dim as usize) {
                for lz in (0..LEAF_DIM).step_by(dim as usize) {
                    let local = [lx, ly, lz];
                    let slot = local_offset(local);
                    let blocked = mask[slot]
                        || is_nonmanifold(&cells.field, add(origin, local), dim)
                        || (use_gradients
                            && !is_mergeable(&gradient, &has_gradient, local, dim, adaptivity));
                    if blocked {
                        mask[local_offset(local.map(|c| c & coord_mask))] = true;
                    } else {
                        for dx in 0..dim {
                            for dy in 0..dim {
                                for dz in 0..dim {
                                    regions[local_offset([lx + dx, ly + dy, lz + dz])] = region_id;
                                }
                            }
                        }
                        region_id += 1;
                    }
                }
            }
        }
        dim <<= 1;
    }
    regions
}

/// `computePoint`: the average zero crossing of one edge group of a cell, in cell-local space.
fn compute_point(values: &[f64; 8], signs: u8, group: u8) -> [f64; 3] {
    let row = &EDGE_GROUPS[usize::from(signs)];
    let zero = |a: usize, b: usize| (0.0 - values[a]) / (values[b] - values[a]);
    let mut avg = [0.0_f64; 3];
    let mut samples = 0_u32;
    if row[1] == group {
        avg[0] += zero(0, 1);
        samples += 1;
    }
    if row[2] == group {
        avg[0] += 1.0;
        avg[2] += zero(1, 2);
        samples += 1;
    }
    if row[3] == group {
        avg[0] += zero(3, 2);
        avg[2] += 1.0;
        samples += 1;
    }
    if row[4] == group {
        avg[2] += zero(0, 3);
        samples += 1;
    }
    if row[5] == group {
        avg[0] += zero(4, 5);
        avg[1] += 1.0;
        samples += 1;
    }
    if row[6] == group {
        avg[0] += 1.0;
        avg[1] += 1.0;
        avg[2] += zero(5, 6);
        samples += 1;
    }
    if row[7] == group {
        avg[0] += zero(7, 6);
        avg[1] += 1.0;
        avg[2] += 1.0;
        samples += 1;
    }
    if row[8] == group {
        avg[1] += 1.0;
        avg[2] += zero(4, 7);
        samples += 1;
    }
    if row[9] == group {
        avg[1] += zero(0, 4);
        samples += 1;
    }
    if row[10] == group {
        avg[0] += 1.0;
        avg[1] += zero(1, 5);
        samples += 1;
    }
    if row[11] == group {
        avg[0] += 1.0;
        avg[1] += zero(2, 6);
        avg[2] += 1.0;
        samples += 1;
    }
    if row[12] == group {
        avg[1] += zero(3, 7);
        avg[2] += 1.0;
        samples += 1;
    }
    if samples > 1 {
        let weight = 1.0 / f64::from(samples);
        avg = avg.map(|component| component * weight);
    }
    avg
}

fn corner_values(field: &Field, origin: [i64; 3]) -> [f64; 8] {
    std::array::from_fn(|corner| f64::from(field.value(add(origin, CORNERS[corner]))))
}

/// `indexToWorld` for a uniform voxel transform, rounded to float32 like `Vec3s`.
fn world_point(index_point: [f64; 3], voxel_size: f64) -> DVec3 {
    DVec3::new(
        f64::from((index_point[0] * voxel_size) as f32),
        f64::from((index_point[1] * voxel_size) as f32),
        f64::from((index_point[2] * voxel_size) as f32),
    )
}

/// `AdaptivePrimBuilder::addPrim`: keeps distinct quads and degenerate quads as triangles.
fn push_primitive(
    quads: &mut Vec<[usize; 4]>,
    triangles: &mut Vec<[usize; 3]>,
    vertices: [usize; 4],
    reverse: bool,
) {
    let [v0, v1, v2, v3] = vertices;
    if v0 != v1 && v0 != v2 && v0 != v3 && v1 != v2 && v1 != v3 && v2 != v3 {
        quads.push(if reverse { [v3, v2, v1, v0] } else { vertices });
    } else if v0 == v3 && v1 != v2 && v1 != v0 && v2 != v0 {
        push_triangle(triangles, [v0, v1, v2], reverse);
    } else if v1 == v2 && v0 != v3 && v0 != v1 && v3 != v1 {
        push_triangle(triangles, [v0, v1, v3], reverse);
    } else if v0 == v1 && v2 != v3 && v2 != v0 && v3 != v0 {
        push_triangle(triangles, [v0, v2, v3], reverse);
    } else if v2 == v3 && v0 != v1 && v0 != v2 && v1 != v2 {
        push_triangle(triangles, [v0, v1, v2], reverse);
    }
}

fn push_triangle(triangles: &mut Vec<[usize; 3]>, [a, b, c]: [usize; 3], reverse: bool) {
    triangles.push(if reverse { [c, b, a] } else { [a, b, c] });
}

/// `ComputePolygons` neighbor lookup: the four vertices of one edge quad, or `None` when any
/// neighbor cell is off (`activeValues` is false in `OpenVDB`).
fn quad_vertices(
    cells: &Cells,
    point_index: &[usize],
    coord: [i64; 3],
    first: i64,
    neighbors: [([i64; 3], usize); 3],
) -> Option<[usize; 4]> {
    let mut vertices = [first, 0, 0, 0];
    for (slot, (offset, edge)) in neighbors.into_iter().enumerate() {
        let index = dense_index(add(coord, offset), cells.field.min_coord, cells.dimensions)?;
        let flag = cells.flags[index];
        if flag == 0 {
            return None;
        }
        let row = &EDGE_GROUPS[usize::from(flag & 0xFF)];
        let group = if row[0] > 1 {
            i64::from(row[edge]) - 1
        } else {
            0
        };
        vertices[slot + 1] = point_index[index] as i64 + group;
    }
    let mut out = [0_usize; 4];
    for (slot, vertex) in vertices.into_iter().enumerate() {
        out[slot] = usize::try_from(vertex).ok()?;
    }
    Some(out)
}

/// Leaf order of `OpenVDB`'s tree traversal: root tiles (4096), then 128-spans, then leaves.
fn leaf_order(origin: [i64; 3]) -> [i64; 9] {
    [
        origin[0] >> 12,
        origin[1] >> 12,
        origin[2] >> 12,
        (origin[0] >> 7) & 31,
        (origin[1] >> 7) & 31,
        (origin[2] >> 7) & 31,
        (origin[0] >> 3) & 15,
        (origin[1] >> 3) & 15,
        (origin[2] >> 3) & 15,
    ]
}

fn collect_leaves(cells: &Cells) -> Vec<Leaf> {
    let mut by_origin: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (index, flag) in cells.flags.iter().enumerate() {
        if *flag != 0 {
            let origin = cells.coord(index).map(|c| c & !(LEAF_DIM - 1));
            by_origin.entry(origin).or_default().push(index);
        }
    }
    let mut leaves: Vec<Leaf> = by_origin
        .into_iter()
        .map(|(origin, mut indices)| {
            indices.sort_by_key(|&index| local_offset(sub(cells.coord(index), origin)));
            Leaf {
                origin,
                cells: indices,
            }
        })
        .collect();
    leaves.sort_by_key(|leaf| leaf_order(leaf.origin));
    leaves
}

/// Adaptive `volumeToMesh` at `adaptivity` (above `OpenVDB`'s 1e-7 threshold).
///
/// `values` is the point grid (`dimensions`) whose lowest corner has absolute index coordinates
/// `min_coord`, and `background` is the value outside the grid (the positive narrow-band width).
/// Returns `None` when `OpenVDB` would produce no vertices or no polygons.
pub(super) fn volume_to_mesh(
    values: &[f32],
    dimensions: [usize; 3],
    min_coord: [i32; 3],
    voxel_size: f32,
    background: f32,
    adaptivity: f32,
) -> Option<Surface> {
    let min_coord = min_coord.map(i64::from);
    let cell_dimensions = dimensions.map(|dimension| dimension - 1);
    let cell_count = cell_dimensions[0] * cell_dimensions[1] * cell_dimensions[2];
    let mut cells = Cells {
        field: Field {
            values,
            dimensions,
            min_coord,
            background,
        },
        dimensions: cell_dimensions,
        flags: vec![0_u16; cell_count],
    };
    for index in 0..cell_count {
        let coord = cells.coord(index);
        let flag = cell_flags(&cells.field, coord);
        cells.flags[index] = flag;
    }

    let leaves = collect_leaves(&cells);
    if leaves.is_empty() {
        return None;
    }
    let use_gradients = adaptivity < 1.0;
    let regions: Vec<[u32; LEAF_CELLS]> = leaves
        .iter()
        .map(|leaf| merge_leaf(&cells, leaf, adaptivity, use_gradients))
        .collect();

    let voxel = f64::from(voxel_size);
    let mut positions = Vec::new();
    let mut point_index = vec![0_usize; cell_count];
    for (leaf, ids) in leaves.iter().zip(&regions) {
        let mut merged: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for &index in &leaf.cells {
            let coord = cells.coord(index);
            let id = ids[local_offset(sub(coord, leaf.origin))];
            if id != 0 {
                merged.entry(id).or_default().push(index);
                continue;
            }
            let signs = (cells.flags[index] & 0xFF) as u8;
            point_index[index] = positions.len();
            let values = corner_values(&cells.field, coord);
            for group in 1..=EDGE_GROUPS[usize::from(signs)][0] {
                let point = compute_point(&values, signs, group);
                positions.push(world_point(
                    [
                        point[0] + coord[0] as f64,
                        point[1] + coord[1] as f64,
                        point[2] + coord[2] as f64,
                    ],
                    voxel,
                ));
            }
        }
        for indices in merged.values() {
            let mut average = [0.0_f64; 3];
            for &index in indices {
                let coord = cells.coord(index);
                let signs = (cells.flags[index] & 0xFF) as u8;
                let first = compute_point(&corner_values(&cells.field, coord), signs, 1);
                for axis in 0..3 {
                    average[axis] += coord[axis] as f64 + first[axis];
                }
            }
            let slot = positions.len();
            for &index in indices {
                point_index[index] = slot;
            }
            if indices.len() > 1 {
                let weight = 1.0 / indices.len() as f64;
                average = average.map(|component| component * weight);
            }
            positions.push(world_point(average, voxel));
        }
    }

    let mut quads = Vec::new();
    let mut triangles = Vec::new();
    for leaf in &leaves {
        for &index in &leaf.cells {
            let flag = cells.flags[index];
            if (flag & EDGES) == 0 {
                continue;
            }
            let coord = cells.coord(index);
            let row = &EDGE_GROUPS[usize::from(flag & 0xFF)];
            let first = point_index[index] as i64;
            let own = if row[0] > 1 {
                [
                    i64::from(row[1]) - 1,
                    i64::from(row[9]) - 1,
                    i64::from(row[4]) - 1,
                ]
            } else {
                [0; 3]
            };
            let inside = (flag & INSIDE) != 0;
            if (flag & XEDGE) != 0
                && let Some(vertices) = quad_vertices(
                    &cells,
                    &point_index,
                    coord,
                    first + own[0],
                    [([0, -1, 0], 5), ([0, -1, -1], 7), ([0, 0, -1], 3)],
                )
            {
                push_primitive(&mut quads, &mut triangles, vertices, inside);
            }
            if (flag & YEDGE) != 0
                && let Some(vertices) = quad_vertices(
                    &cells,
                    &point_index,
                    coord,
                    first + own[1],
                    [([0, 0, -1], 12), ([-1, 0, -1], 11), ([-1, 0, 0], 10)],
                )
            {
                push_primitive(&mut quads, &mut triangles, vertices, inside);
            }
            if (flag & ZEDGE) != 0
                && let Some(vertices) = quad_vertices(
                    &cells,
                    &point_index,
                    coord,
                    first + own[2],
                    [([0, -1, 0], 8), ([-1, -1, 0], 6), ([-1, 0, 0], 2)],
                )
            {
                push_primitive(&mut quads, &mut triangles, vertices, !inside);
            }
        }
    }

    if positions.is_empty() || quads.len() + triangles.len() == 0 {
        return None;
    }
    Some(Surface {
        positions,
        quads,
        triangles,
    })
}
