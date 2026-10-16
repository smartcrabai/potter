//! Exact angular sort of triangles around an edge (Blender `sort_tris_class`,
//! `sort_by_signed_triangle_index`, `sort_tris_around_edge`).

use super::super::exact::{Rat3, cross3, dot3, sgn, sub3};
use super::super::imesh::{Arena, Tri};
use super::invariant;
use super::topology::{Edge, find_flap_vert};
use crate::error::Result;

/// Blender `EXTRA_TRI_INDEX`: stands for the dummy triangle of [`TriView::extra`].
pub(super) const EXTRA_TRI: usize = usize::MAX;

/// Sort key of [`EXTRA_TRI`] (Blender uses `INT_MAX`).
const EXTRA_SORT_KEY: i64 = i32::MAX as i64;

/// Triangle lookup for sorting: the intersected triangles plus an optional dummy triangle that is
/// addressed as [`EXTRA_TRI`].
pub(super) struct TriView<'a> {
    pub(super) arena: &'a Arena,
    pub(super) tris: &'a [Tri],
    pub(super) extra: Option<Tri>,
}

impl TriView<'_> {
    fn tri(&self, t: usize) -> Result<Tri> {
        if t == EXTRA_TRI {
            return self
                .extra
                .ok_or_else(|| invariant("dummy triangle missing during edge sort"));
        }
        self.tris
            .get(t)
            .copied()
            .ok_or_else(|| invariant("triangle index out of range during edge sort"))
    }

    fn co(&self, v: u32) -> &Rat3 {
        &self.arena.vert(v).co
    }
}

/// Exact orientation of `d` relative to the plane of `a, b, c` (Shewchuk `orient3d`): positive
/// when `d` is below the plane, where `a, b, c` are counter-clockwise seen from above.
fn orient3d_exact(a: &Rat3, b: &Rat3, c: &Rat3, d: &Rat3) -> i8 {
    sgn(&dot3(&sub3(a, d), &cross3(&sub3(b, d), &sub3(c, d))))
}

/// Blender's group numbers 1..=4 from `sort_tris_class`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Group {
    /// Coplanar with the reference triangle, on the same side of the edge.
    SameSide,
    /// Coplanar with the reference triangle, on the opposite side of the edge.
    OppositeSide,
    /// Below the reference plane (non-reversed orientation).
    Below,
    /// Above the reference plane (non-reversed orientation).
    Above,
}

/// Blender `sort_tris_class`: classify `tri` with respect to `tri0`, which shares edge `e`.
fn sort_tris_class(view: &TriView<'_>, tri: Tri, tri0: Tri, e: Edge) -> Result<Group> {
    let (flapv0, rev0) = find_flap_vert(&tri0, e)
        .ok_or_else(|| invariant("reference triangle does not contain the sort edge"))?;
    let (flapv, _) = find_flap_vert(&tri, e)
        .ok_or_else(|| invariant("triangle does not contain the sort edge"))?;
    let orient = orient3d_exact(
        view.co(tri0.v[0]),
        view.co(tri0.v[1]),
        view.co(tri0.v[2]),
        view.co(flapv),
    );
    Ok(if orient > 0 {
        if rev0 { Group::Above } else { Group::Below }
    } else if orient < 0 {
        if rev0 { Group::Below } else { Group::Above }
    } else if flapv == flapv0 {
        Group::SameSide
    } else {
        Group::OppositeSide
    })
}

/// Blender sort key of a triangle index: the index itself, or [`EXTRA_SORT_KEY`].
fn sort_key(t: usize) -> i64 {
    if t == EXTRA_TRI {
        EXTRA_SORT_KEY
    } else {
        t as i64
    }
}

/// Inverse of [`sort_key`] on absolute values.
fn tri_from_sort_key(key: i64) -> usize {
    if key == EXTRA_SORT_KEY {
        EXTRA_TRI
    } else {
        key as usize
    }
}

/// Blender `sort_by_signed_triangle_index`: sort a group by triangle index, with the sign telling
/// whether the triangle traverses `e` reversed. This makes co-planar ties deterministic.
fn sort_by_signed_triangle_index(view: &TriView<'_>, g: &mut [usize], e: Edge) -> Result<()> {
    let mut signed = g
        .iter()
        .map(|&t| -> Result<i64> {
            let tri = view.tri(t)?;
            let rev = find_flap_vert(&tri, e).is_some_and(|(_, rev)| rev);
            let key = sort_key(t);
            Ok(if rev { -key } else { key })
        })
        .collect::<Result<Vec<i64>>>()?;
    signed.sort_unstable();
    for (slot, s) in g.iter_mut().zip(signed) {
        *slot = tri_from_sort_key(s.abs());
    }
    Ok(())
}

/// Blender `sort_tris_around_edge`: order `tris`, which all share edge `e`, as they appear
/// clockwise when looking down `e`. `t0` is the reference of the top-level call; its group merge
/// order differs from the recursive calls.
pub(super) fn sort_tris_around_edge(
    view: &TriView<'_>,
    e: Edge,
    tris: &[usize],
    t0: usize,
) -> Result<Vec<usize>> {
    let Some((&first, rest)) = tris.split_first() else {
        return Ok(Vec::new());
    };
    let triref = view.tri(first)?;
    let mut g1 = vec![first];
    let mut g2 = Vec::new();
    let mut g3 = Vec::new();
    let mut g4 = Vec::new();
    for &t in rest {
        let group = sort_tris_class(view, view.tri(t)?, triref, e)?;
        match group {
            Group::SameSide => g1.push(t),
            Group::OppositeSide => g2.push(t),
            Group::Below => g3.push(t),
            Group::Above => g4.push(t),
        }
    }
    if g1.len() > 1 {
        sort_by_signed_triangle_index(view, &mut g1, e)?;
    }
    if g2.len() > 1 {
        sort_by_signed_triangle_index(view, &mut g2, e)?;
    }
    if g3.len() > 1 {
        g3 = sort_tris_around_edge(view, e, &g3, t0)?;
    }
    if g4.len() > 1 {
        g4 = sort_tris_around_edge(view, e, &g4, t0)?;
    }
    let mut ans = Vec::with_capacity(tris.len());
    if first == t0 {
        ans.extend(g1);
        ans.extend(g4);
        ans.extend(g2);
        ans.extend(g3);
    } else {
        ans.extend(g3);
        ans.extend(g1);
        ans.extend(g4);
        ans.extend(g2);
    }
    Ok(ans)
}
