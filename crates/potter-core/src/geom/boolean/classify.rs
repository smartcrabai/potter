//! Classification of intersected triangles (Blender `boolean_trimesh`).
//!
//! PWN meshes use patches, cells, exact angular sorts around edges, winding propagation from an
//! ambient cell, and extraction of the cut triangles. Non-PWN meshes use the float-ray fallback.

mod ambient;
mod patches;
#[path = "raycast.rs"]
mod raycast;
mod sort;
#[cfg(test)]
mod tests;
mod topology;

use super::imesh::{Arena, Tri};
use super::intersect::trimesh_nary_intersect;
use crate::error::{ErrorCode, PotError, Result};
use ambient::{find_ambient_cell, finish_patch_cell_graph};
use patches::{CellsInfo, PatchesInfo, Side, find_cells, find_patches, patch_cell_graph_ok};
use topology::{Topology, is_pwn};

/// Blender `BoolOpType` for the EXACT solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BoolOp {
    Intersect,
    Union,
    Difference,
}

/// Internal invariant violation of the classification (a bug or malformed intersection output).
fn invariant(what: &str) -> PotError {
    PotError::new(
        ErrorCode::InternalError,
        format!("boolean classification invariant violated: {what}"),
    )
}

/// Blender `boolean_trimesh`: intersects `tris`, then classifies the result by the PWN cell
/// graph or, for non-PWN meshes, Blender's float-ray fallback.
///
/// `shape_fn` maps a triangle's `orig` to its operand index in `0..nshapes`. `hole_tolerant`
/// selects per-triangle raycasting instead of per-patch classification for non-PWN meshes.
pub(crate) fn boolean_tris(
    arena: &mut Arena,
    tris: &[Tri],
    nshapes: usize,
    shape_fn: &dyn Fn(i32) -> usize,
    op: BoolOp,
    use_self: bool,
    hole_tolerant: bool,
) -> Result<Vec<Tri>> {
    let tm_si = trimesh_nary_intersect(arena, tris, nshapes, shape_fn, use_self);
    if tm_si.is_empty() {
        return Ok(tm_si);
    }
    let topo = Topology::new(&tm_si);
    if !is_pwn(&tm_si, &topo) {
        return Ok(raycast::raycast_boolean(
            arena,
            &tm_si,
            nshapes,
            shape_fn,
            op,
            hole_tolerant,
        ));
    }
    let mut pinfo = find_patches(&tm_si, &topo);
    let mut cinfo = find_cells(arena, &tm_si, &topo, &mut pinfo)?;
    finish_patch_cell_graph(arena, &tm_si, &topo, &mut cinfo, &mut pinfo)?;
    if !patch_cell_graph_ok(&cinfo, &pinfo) {
        return Err(invariant(
            "patch/cell graph is disconnected; the boolean input is not valid",
        ));
    }
    cinfo.init_windings(nshapes);
    let Some(c_ambient) = find_ambient_cell(arena, &tm_si, &topo, &pinfo, None)? else {
        return Err(invariant(
            "no ambient cell found; the boolean input is not valid",
        ));
    };
    propagate_windings_and_in_output_volume(&pinfo, &mut cinfo, c_ambient, op, nshapes, &|t| {
        shape_fn(tm_si[t].orig)
    })?;
    extract_from_in_output_volume_diffs(&tm_si, &pinfo, &cinfo)
}

/// Blender `apply_bool_op`: whether a cell with these windings (one per operand) is in the output.
///
/// INTERSECT: every winding is nonzero. UNION: some winding is nonzero. DIFFERENCE: winding 0 is
/// nonzero and no other winding is positive.
#[must_use]
pub(crate) fn apply_bool_op(op: BoolOp, winding: &[i32]) -> bool {
    match op {
        BoolOp::Intersect => winding.iter().all(|&w| w != 0),
        BoolOp::Union => winding.iter().any(|&w| w != 0),
        BoolOp::Difference => match winding.split_first() {
            Some((&first, rest)) if first != 0 => rest.iter().all(|&w| w < 1),
            _ => false,
        },
    }
}

/// Blender `propagate_windings_and_in_output_volume`: breadth-first from the ambient cell. Crossing
/// a patch changes its operand's winding by one, and sets the cell's output membership.
fn propagate_windings_and_in_output_volume(
    pinfo: &PatchesInfo,
    cinfo: &mut CellsInfo,
    c_ambient: usize,
    op: BoolOp,
    nshapes: usize,
    shape_of: &dyn Fn(usize) -> usize,
) -> Result<()> {
    let ambient = cinfo
        .cells
        .get_mut(c_ambient)
        .ok_or_else(|| invariant("ambient cell index out of range"))?;
    ambient.winding = vec![0; nshapes];
    ambient.winding_assigned = true;
    let mut queue = vec![c_ambient];
    let mut head = 0;
    while head < queue.len() {
        let c = queue[head];
        head += 1;
        let from_winding = cinfo.cells[c].winding.clone();
        let patches: Vec<usize> = cinfo.cells[c].patches.iter().copied().collect();
        for p in patches {
            let patch = &pinfo.patches[p];
            let p_above_c = patch.cell_below == Some(c);
            let neighbor = if p_above_c {
                patch.cell_above
            } else {
                patch.cell_below
            }
            .ok_or_else(|| invariant("patch without a neighbouring cell"))?;
            if cinfo.cells[neighbor].winding_assigned {
                continue;
            }
            let delta = if p_above_c { -1 } else { 1 };
            let &t = patch
                .tris
                .first()
                .ok_or_else(|| invariant("patch without triangles"))?;
            let shape = shape_of(t);
            let mut winding = from_winding.clone();
            let slot = winding
                .get_mut(shape)
                .ok_or_else(|| invariant("triangle operand index exceeds operand count"))?;
            *slot += delta;
            let cell = &mut cinfo.cells[neighbor];
            cell.in_output_volume = apply_bool_op(op, &winding);
            cell.winding = winding;
            cell.winding_assigned = true;
            queue.push(neighbor);
        }
    }
    Ok(())
}

/// Blender `Face` flip used for output: vertices reversed, edge origins and intersect flags
/// reversed along with them.
#[must_use]
fn flipped(tri: Tri) -> Tri {
    Tri {
        v: [tri.v[0], tri.v[2], tri.v[1]],
        orig: tri.orig,
        edge_orig: [tri.edge_orig[2], tri.edge_orig[1], tri.edge_orig[0]],
        is_intersect: [
            tri.is_intersect[2],
            tri.is_intersect[1],
            tri.is_intersect[0],
        ],
    }
}

fn cell_at(cinfo: &CellsInfo, c: Option<usize>) -> Result<&patches::Cell> {
    let c = c.ok_or_else(|| invariant("patch without a cell"))?;
    cinfo
        .cells
        .get(c)
        .ok_or_else(|| invariant("cell index out of range"))
}

/// Blender `extract_zero_volume_cell_tris`: for stacks of identical triangles that enclose zero
/// volume, emit one representative when the cells on either side of the stack differ in output
/// membership.
fn extract_zero_volume_cell_tris(
    out: &mut Vec<Tri>,
    tris: &[Tri],
    pinfo: &PatchesInfo,
    cinfo: &CellsInfo,
) -> Result<()> {
    let n = pinfo.patches.len();
    let mut adj_to_zv = vec![false; n];
    for (p, patch) in pinfo.patches.iter().enumerate() {
        adj_to_zv[p] = cell_at(cinfo, patch.cell_above)?.zero_volume
            || cell_at(cinfo, patch.cell_below)?.zero_volume;
    }
    let mut allocated = vec![false; n];
    for p in 0..n {
        if !adj_to_zv[p] || allocated[p] {
            continue;
        }
        let mut stack = vec![p];
        let mut flipped_flags = vec![false];
        allocated[p] = true;
        // Walk from p through zero-volume cells on its above side, then its below side.
        let above_stack_cell = walk_zero_volume(
            pinfo,
            cinfo,
            p,
            Side::Above,
            &mut stack,
            &mut flipped_flags,
            &mut allocated,
        )?;
        let below_stack_cell = walk_zero_volume(
            pinfo,
            cinfo,
            p,
            Side::Below,
            &mut stack,
            &mut flipped_flags,
            &mut allocated,
        )?;
        let above_in = above_stack_cell.in_output_volume;
        if above_in ^ below_stack_cell.in_output_volume {
            let need_flipped = above_in;
            if let Some(i) = (0..stack.len()).find(|&i| flipped_flags[i] == need_flipped) {
                let t = *pinfo.patches[stack[i]]
                    .tris
                    .first()
                    .ok_or_else(|| invariant("patch without triangles"))?;
                out.push(tris[t]);
            } else {
                let t = *pinfo.patches[p]
                    .tris
                    .first()
                    .ok_or_else(|| invariant("patch without triangles"))?;
                out.push(flipped(tris[t]));
            }
        }
    }
    Ok(())
}

/// Walk from patch `p` across zero-volume cells on `side`, appending each patch of the stack and
/// its orientation relative to `p`. Returns the first non-zero-volume cell reached.
///
/// Blender's walk loops on zero-volume cells without a bound; the guard below stops a closed loop
/// of zero-volume cells (no bounding cell) with an error instead of hanging.
fn walk_zero_volume<'c>(
    pinfo: &PatchesInfo,
    cinfo: &'c CellsInfo,
    p: usize,
    side: Side,
    stack: &mut Vec<usize>,
    flipped_flags: &mut Vec<bool>,
    allocated: &mut [bool],
) -> Result<&'c patches::Cell> {
    let mut pwalk = p;
    let mut c = pinfo.patches[pwalk].cell(side);
    let mut cell = cell_at(cinfo, c)?;
    let mut steps = 0;
    while cell.zero_volume {
        steps += 1;
        if steps > pinfo.patches.len() {
            return Err(invariant("closed loop of zero-volume cells"));
        }
        let pother = cell
            .patch_other(pwalk)
            .ok_or_else(|| invariant("zero-volume cell without two patches"))?;
        // `flip`: the other patch lists the walked cell on the same side as `side`.
        let flip = pinfo.patches[pother].cell(side) == c;
        flipped_flags.push(flip);
        stack.push(pother);
        allocated[pother] = true;
        pwalk = pother;
        let next_side = if flip { other_side(side) } else { side };
        c = pinfo.patches[pwalk].cell(next_side);
        cell = cell_at(cinfo, c)?;
    }
    Ok(cell)
}

fn other_side(side: Side) -> Side {
    match side {
        Side::Above => Side::Below,
        Side::Below => Side::Above,
    }
}

/// Blender `extract_from_in_output_volume_diffs`: keep each triangle whose two cells differ in
/// output membership, flipped so that its normal points out of the output volume. Zero-volume
/// stacks are handled separately.
fn extract_from_in_output_volume_diffs(
    tris: &[Tri],
    pinfo: &PatchesInfo,
    cinfo: &CellsInfo,
) -> Result<Vec<Tri>> {
    let mut out = Vec::new();
    let mut any_zero_volume_cell = false;
    for (t, tri) in tris.iter().enumerate() {
        let p = pinfo.tri_patch[t].ok_or_else(|| invariant("triangle without patch"))?;
        let patch = &pinfo.patches[p];
        let above = cell_at(cinfo, patch.cell_above)?;
        let below = cell_at(cinfo, patch.cell_below)?;
        let adjacent_zero_volume_cell = above.zero_volume || below.zero_volume;
        any_zero_volume_cell |= adjacent_zero_volume_cell;
        if (above.in_output_volume ^ below.in_output_volume) && !adjacent_zero_volume_cell {
            if above.in_output_volume {
                out.push(flipped(*tri));
            } else {
                out.push(*tri);
            }
        }
    }
    if any_zero_volume_cell {
        extract_zero_volume_cell_tris(&mut out, tris, pinfo, cinfo)?;
    }
    Ok(out)
}
