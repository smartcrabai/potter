//! Patches (triangles joined across manifold edges), cells (volumes between patches), and the
//! patch-cell bipartite graph built by sorting triangles around each shared edge. Blender
//! `find_patches`, `find_cells`, `find_cells_from_edge`, `merge_cells`, `find_patch_components`,
//! `patch_cell_graph_ok`.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use super::super::imesh::{Arena, Tri};
use super::invariant;
use super::sort::{TriView, sort_tris_around_edge};
use super::topology::{Edge, Topology, find_flap_vert};
use crate::error::Result;

/// Side of a patch: the cell the triangle normal points into (`Above`) or away from (`Below`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Side {
    Above,
    Below,
}

/// Blender `Patch`: a maximal set of triangles connected through manifold edges.
#[derive(Debug, Default)]
pub(super) struct Patch {
    /// Triangle indices. The first is the seed.
    pub(super) tris: Vec<usize>,
    pub(super) cell_above: Option<usize>,
    pub(super) cell_below: Option<usize>,
    pub(super) component: Option<usize>,
}

impl Patch {
    #[must_use]
    pub(super) fn cell(&self, side: Side) -> Option<usize> {
        match side {
            Side::Above => self.cell_above,
            Side::Below => self.cell_below,
        }
    }

    fn set_cell(&mut self, side: Side, cell: Option<usize>) {
        match side {
            Side::Above => self.cell_above = cell,
            Side::Below => self.cell_below = cell,
        }
    }
}

/// Blender `PatchesInfo`.
#[derive(Debug, Default)]
pub(super) struct PatchesInfo {
    pub(super) patches: Vec<Patch>,
    /// Patch of each triangle.
    pub(super) tri_patch: Vec<Option<usize>>,
    /// Shared edge of each pair of incident patches, stored under both orders.
    pub(super) pp_edge: BTreeMap<(usize, usize), Edge>,
}

impl PatchesInfo {
    fn add_patch(&mut self) -> usize {
        self.patches.push(Patch::default());
        self.patches.len() - 1
    }

    fn grow_patch(&mut self, patch: usize, t: usize) {
        self.tri_patch[t] = Some(patch);
        self.patches[patch].tris.push(t);
    }

    #[must_use]
    fn patch_patch_edge(&self, p1: usize, p2: usize) -> Option<Edge> {
        self.pp_edge.get(&(p1, p2)).copied()
    }

    fn add_new_patch_patch_edge(&mut self, p1: usize, p2: usize, e: Edge) {
        self.pp_edge.insert((p1, p2), e);
        self.pp_edge.insert((p2, p1), e);
    }
}

/// Blender `Cell`: a volume bounded by patches. Its patches are a set.
#[derive(Debug, Default)]
pub(super) struct Cell {
    pub(super) patches: BTreeSet<usize>,
    pub(super) winding: Vec<i32>,
    pub(super) merged_to: Option<usize>,
    pub(super) winding_assigned: bool,
    pub(super) in_output_volume: bool,
    /// Set when the cell is bounded by exactly two identical single-triangle patches.
    pub(super) zero_volume: bool,
}

impl Cell {
    fn add_patch(&mut self, p: usize) {
        self.patches.insert(p);
        self.zero_volume = false;
    }

    /// In a cell with exactly two patches, the patch that is not `p`.
    #[must_use]
    pub(super) fn patch_other(&self, p: usize) -> Option<usize> {
        if self.patches.len() != 2 {
            return None;
        }
        self.patches.iter().copied().find(|&q| q != p)
    }

    /// Blender `Cell::check_for_zero_volume`: a cell between two identical triangles has zero
    /// volume.
    fn check_for_zero_volume(&mut self, pinfo: &PatchesInfo, tris: &[Tri]) {
        if self.patches.len() != 2 {
            return;
        }
        let mut it = self.patches.iter().copied();
        let (Some(p1), Some(p2)) = (it.next(), it.next()) else {
            return;
        };
        let (patch1, patch2) = (&pinfo.patches[p1], &pinfo.patches[p2]);
        if patch1.tris.len() == 1
            && patch2.tris.len() == 1
            && tris_have_same_verts(&tris[patch1.tris[0]], &tris[patch2.tris[0]])
        {
            self.zero_volume = true;
        }
    }
}

/// Blender `tris_have_same_verts`: same three vertices in either cyclic order.
fn tris_have_same_verts(a: &Tri, b: &Tri) -> bool {
    let (a, b) = (a.v, b.v);
    if a[0] == b[0] {
        (a[1] == b[1] && a[2] == b[2]) || (a[1] == b[2] && a[2] == b[1])
    } else if a[0] == b[1] {
        (a[1] == b[0] && a[2] == b[2]) || (a[1] == b[2] && a[2] == b[0])
    } else if a[0] == b[2] {
        (a[1] == b[0] && a[2] == b[1]) || (a[1] == b[1] && a[2] == b[0])
    } else {
        false
    }
}

/// Blender `CellsInfo`.
#[derive(Debug, Default)]
pub(super) struct CellsInfo {
    pub(super) cells: Vec<Cell>,
}

impl CellsInfo {
    fn add_cell(&mut self) -> usize {
        self.cells.push(Cell::default());
        self.cells.len() - 1
    }

    pub(super) fn init_windings(&mut self, nshapes: usize) {
        for cell in &mut self.cells {
            cell.winding = vec![0; nshapes];
        }
    }
}

/// Blender `find_patches`: grow patches across manifold edges, and record patch-patch edges where
/// a non-manifold edge separates two patches.
#[must_use]
pub(super) fn find_patches(tris: &[Tri], topo: &Topology) -> PatchesInfo {
    let mut pinfo = PatchesInfo {
        patches: Vec::new(),
        tri_patch: vec![None; tris.len()],
        pp_edge: BTreeMap::new(),
    };
    let t_others: Vec<[Option<usize>; 3]> = tris
        .iter()
        .enumerate()
        .map(|(t, tri)| {
            std::array::from_fn(|i| {
                topo.other_tri_if_manifold(Edge::new(tri.v[i], tri.v[(i + 1) % 3]), t)
            })
        })
        .collect();
    for t in 0..tris.len() {
        if pinfo.tri_patch[t].is_some() {
            continue;
        }
        let cur = pinfo.add_patch();
        let mut grow = vec![t];
        while let Some(tcand) = grow.pop() {
            if pinfo.tri_patch[tcand].is_some() {
                continue;
            }
            pinfo.grow_patch(cur, tcand);
            let tri = tris[tcand];
            for (i, other) in t_others[tcand].into_iter().enumerate() {
                let e = Edge::new(tri.v[i], tri.v[(i + 1) % 3]);
                if let Some(t_other) = other {
                    if pinfo.tri_patch[t_other].is_none() {
                        grow.push(t_other);
                    }
                } else {
                    let incident = topo.edge_tris(e).unwrap_or_default();
                    for &t_other in incident {
                        if t_other == tcand {
                            continue;
                        }
                        let Some(p_other) = pinfo.tri_patch[t_other] else {
                            continue;
                        };
                        if p_other != cur && pinfo.patch_patch_edge(cur, p_other).is_none() {
                            pinfo.add_new_patch_patch_edge(cur, p_other, e);
                        }
                    }
                }
            }
        }
    }
    pinfo
}

/// Blender `merge_cells`: move `merge_from`'s patches into the root of `merge_to`.
pub(super) fn merge_cells(
    cinfo: &mut CellsInfo,
    pinfo: &mut PatchesInfo,
    merge_to: usize,
    merge_from: usize,
) {
    if merge_to == merge_from {
        return;
    }
    let mut root = merge_to;
    while let Some(next) = cinfo.cells[root].merged_to {
        root = next;
    }
    // shortcut: Blender would point `merge_from` at its own root and loop forever in the
    // `merged_to` chase; a merge into the same root is a no-op.
    if root == merge_from {
        return;
    }
    let from_patches: Vec<usize> = cinfo.cells[merge_from].patches.iter().copied().collect();
    for p in from_patches {
        cinfo.cells[root].add_patch(p);
        let patch = &mut pinfo.patches[p];
        if patch.cell_above == Some(merge_from) {
            patch.cell_above = Some(merge_to);
        }
        if patch.cell_below == Some(merge_from) {
            patch.cell_below = Some(merge_to);
        }
    }
    cinfo.cells[merge_from].merged_to = Some(root);
}

/// Blender `find_cells_from_edge`: sweep the triangles around one patch-patch edge and link each
/// adjacent pair of patches through a shared cell.
fn find_cells_from_edge(
    arena: &Arena,
    tris: &[Tri],
    topo: &Topology,
    pinfo: &mut PatchesInfo,
    cinfo: &mut CellsInfo,
    e: Edge,
) -> Result<()> {
    let etris = topo
        .edge_tris(e)
        .ok_or_else(|| invariant("patch-patch edge without triangles"))?;
    let &t0 = etris
        .first()
        .ok_or_else(|| invariant("patch-patch edge without triangles"))?;
    let view = TriView {
        arena,
        tris,
        extra: None,
    };
    let sorted = sort_tris_around_edge(&view, e, etris, t0)?;
    let n = sorted.len();
    let edge_patches = sorted
        .iter()
        .map(|&t| {
            pinfo.tri_patch[t].ok_or_else(|| invariant("triangle without patch during cell sweep"))
        })
        .collect::<Result<Vec<usize>>>()?;
    for i in 0..n {
        let inext = (i + 1) % n;
        let r_index = edge_patches[i];
        let rnext_index = edge_patches[inext];
        let (_, r_flipped) = find_flap_vert(&tris[sorted[i]], e)
            .ok_or_else(|| invariant("sorted triangle lost the sort edge"))?;
        let (_, rnext_flipped) = find_flap_vert(&tris[sorted[inext]], e)
            .ok_or_else(|| invariant("sorted triangle lost the sort edge"))?;
        let r_side = if r_flipped { Side::Below } else { Side::Above };
        let rnext_side = if rnext_flipped {
            Side::Above
        } else {
            Side::Below
        };
        let r_follow = pinfo.patches[r_index].cell(r_side);
        let rnext_prev = pinfo.patches[rnext_index].cell(rnext_side);
        match (r_follow, rnext_prev) {
            (None, None) => {
                let c = cinfo.add_cell();
                pinfo.patches[r_index].set_cell(r_side, Some(c));
                pinfo.patches[rnext_index].set_cell(rnext_side, Some(c));
                cinfo.cells[c].add_patch(r_index);
                cinfo.cells[c].add_patch(rnext_index);
                cinfo.cells[c].check_for_zero_volume(pinfo, tris);
            }
            (Some(c), None) => {
                pinfo.patches[rnext_index].set_cell(rnext_side, Some(c));
                cinfo.cells[c].add_patch(rnext_index);
                cinfo.cells[c].check_for_zero_volume(pinfo, tris);
            }
            (None, Some(c)) => {
                pinfo.patches[r_index].set_cell(r_side, Some(c));
                cinfo.cells[c].add_patch(r_index);
                cinfo.cells[c].check_for_zero_volume(pinfo, tris);
            }
            (Some(follow), Some(prev)) => {
                // Blender merges only when the follow cell has at least as many patches; the
                // same-cell case is a no-op.
                if follow != prev
                    && cinfo.cells[follow].patches.len() >= cinfo.cells[prev].patches.len()
                {
                    merge_cells(cinfo, pinfo, follow, prev);
                }
            }
        }
    }
    Ok(())
}

/// Blender `find_cells`: build the cells around every patch-patch edge, then give every patch
/// that still lacks a cell an isolated one.
pub(super) fn find_cells(
    arena: &Arena,
    tris: &[Tri],
    topo: &Topology,
    pinfo: &mut PatchesInfo,
) -> Result<CellsInfo> {
    let mut cinfo = CellsInfo::default();
    let mut processed: HashSet<Edge> = HashSet::new();
    let pairs: Vec<((usize, usize), Edge)> = pinfo.pp_edge.iter().map(|(&k, &e)| (k, e)).collect();
    for ((p, q), e) in pairs {
        if p < q && processed.insert(e) {
            find_cells_from_edge(arena, tris, topo, pinfo, &mut cinfo, e)?;
        }
    }
    for p in 0..pinfo.patches.len() {
        if pinfo.patches[p].cell_above.is_none() {
            let c = cinfo.add_cell();
            pinfo.patches[p].cell_above = Some(c);
            cinfo.cells[c].add_patch(p);
        }
        if pinfo.patches[p].cell_below.is_none() {
            let c = cinfo.add_cell();
            pinfo.patches[p].cell_below = Some(c);
            cinfo.cells[c].add_patch(p);
        }
    }
    Ok(cinfo)
}

/// Blender `find_patch_components`: connected components of patches through shared cells. Sets
/// each patch's `component` and returns the patches of every component.
pub(super) fn find_patch_components(cinfo: &CellsInfo, pinfo: &mut PatchesInfo) -> Vec<Vec<usize>> {
    let mut cell_processed = vec![false; cinfo.cells.len()];
    let mut components: Vec<Vec<usize>> = Vec::new();
    for pstart in 0..pinfo.patches.len() {
        if pinfo.patches[pstart].component.is_some() {
            continue;
        }
        let comp = components.len();
        components.push(vec![pstart]);
        pinfo.patches[pstart].component = Some(comp);
        let mut stack = vec![pstart];
        while let Some(p) = stack.pop() {
            for c in [pinfo.patches[p].cell_above, pinfo.patches[p].cell_below]
                .into_iter()
                .flatten()
            {
                if cell_processed[c] {
                    continue;
                }
                cell_processed[c] = true;
                for &pn in &cinfo.cells[c].patches {
                    if pinfo.patches[pn].component.is_none() {
                        pinfo.patches[pn].component = Some(comp);
                        stack.push(pn);
                        components[comp].push(pn);
                    }
                }
            }
        }
    }
    components
}

/// Blender `patch_cell_graph_ok`: every live cell has patches, and every patch has two cells.
#[must_use]
pub(super) fn patch_cell_graph_ok(cinfo: &CellsInfo, pinfo: &PatchesInfo) -> bool {
    for cell in &cinfo.cells {
        if cell.merged_to.is_some() {
            continue;
        }
        if cell.patches.is_empty() || cell.patches.iter().any(|&p| p >= pinfo.patches.len()) {
            return false;
        }
    }
    pinfo
        .patches
        .iter()
        .all(|patch| match (patch.cell_above, patch.cell_below) {
            (Some(a), Some(b)) => a < cinfo.cells.len() && b < cinfo.cells.len(),
            _ => false,
        })
}
