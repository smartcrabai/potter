//! Triangle-triangle intersection (Blender `intersect_tri_tri`, `mesh_intersect.cc`).
//!
//! Guigue-Devillers classification ("Faster Triangle-Triangle Intersection Tests"), exact.
//! Each triangle's vertices are first classified against the other triangle's plane with the
//! double filter `filter_plane_side`; only undecided signs (0) are re-tested exactly. The
//! canonicalisation of the vertex order and the chart in `itt_canon2` are Blender's.
//!
//! Plane normals are the exact `cross(p - r, q - r)` of each triangle, as Blender's
//! `Face::populate_plane(true)`. The double copy used by the filters is the nearest double of that
//! exact normal (Blender truncates via `mpq_class::get_d`; both roundings satisfy the filter's
//! error bound). Planes are recomputed per call and not cached, since the contract passes two
//! triangles only.
//!
//! Deviations from Blender:
//! - `ITT_value::t_source` is not returned. Blender stores the second triangle for coplanar
//!   results; the caller already holds that index.
//! - Coplanar pairs return [`IttKind::Coplanar`] without a 2D overlap test, as in Blender. Whether
//!   they overlap is decided by the coplanar-cluster stage, not here.

use glam::DVec3;
use num_traits::Zero;

use super::super::exact::{
    Rat, Rat3, cross3, dot3, filter_plane_side, plane_side_exact, rat_to_f64, scale3, sgn, sub3,
};
use super::super::imesh::{Arena, Tri};

/// Blender `ITT_value_kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IttKind {
    /// Blender `INONE`: the triangles do not intersect.
    None,
    /// Blender `IPOINT`: a single intersection point `p1`.
    Point,
    /// Blender `ISEGMENT`: an intersection segment `p1`..`p2`.
    Segment,
    /// Blender `ICOPLANAR`: the triangles lie in one plane.
    Coplanar,
}

/// Blender `ITT_value`: the intersection of two triangles.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Itt {
    pub kind: IttKind,
    /// Valid for `Point` and `Segment`; zero otherwise.
    pub p1: Rat3,
    /// Valid for `Segment`; zero otherwise.
    pub p2: Rat3,
}

impl Itt {
    fn none() -> Self {
        Self {
            kind: IttKind::None,
            p1: zero3(),
            p2: zero3(),
        }
    }

    fn point(p: Rat3) -> Self {
        Self {
            kind: IttKind::Point,
            p1: p,
            p2: zero3(),
        }
    }

    fn segment(p1: Rat3, p2: Rat3) -> Self {
        Self {
            kind: IttKind::Segment,
            p1,
            p2,
        }
    }

    fn coplanar() -> Self {
        Self {
            kind: IttKind::Coplanar,
            p1: zero3(),
            p2: zero3(),
        }
    }
}

fn zero3() -> Rat3 {
    [Rat::zero(), Rat::zero(), Rat::zero()]
}

/// Exact normal of triangle `(p, q, r)`: `cross(p - r, q - r)`, as Blender's `populate_plane`.
fn tri_normal(p: &Rat3, q: &Rat3, r: &Rat3) -> Rat3 {
    cross3(&sub3(p, r), &sub3(q, r))
}

/// Double copy of an exact plane normal, used only by the float filters (Blender `Plane::norm`).
fn normal_f(n: &Rat3) -> DVec3 {
    DVec3::new(rat_to_f64(&n[0]), rat_to_f64(&n[1]), rat_to_f64(&n[2]))
}

/// True when all three signs are nonzero and equal: the triangle lies strictly on one side.
fn same_strict_side(s: [i8; 3]) -> bool {
    s.iter().all(|&x| x > 0) || s.iter().all(|&x| x < 0)
}

/// Blender `tti_interp`: the point where line `ab` meets the plane through `c` with normal `n`.
/// Requires `dot(a - b, n) != 0`, which the classification guarantees at every call site.
fn tti_interp(a: &Rat3, b: &Rat3, c: &Rat3, n: &Rat3) -> Rat3 {
    let ab = sub3(a, b);
    let ac = sub3(a, c);
    let den = dot3(&ab, n);
    debug_assert!(!den.is_zero(), "tti_interp: segment parallel to plane");
    let alpha = dot3(&ac, n) / den;
    sub3(a, &scale3(&alpha, &ab))
}

/// Blender `tti_above`: `sgn(dot(ad, n))` with `n = cross(b - a, c - a)`. Gives +1, 0, -1 as
/// `a + ad` is above, on, or below the CCW-oriented plane through `a`, `b`, `c`.
fn tti_above(a: &Rat3, b: &Rat3, c: &Rat3, ad: &Rat3) -> i8 {
    let n = cross3(&sub3(b, a), &sub3(c, a));
    sgn(&dot3(ad, &n))
}

/// Blender `itt_canon2`. Vertices are in canonical order: `p1` is off the second plane while
/// `q1`, `r1` are on the plane or on the other side; `p2` likewise for the first plane.
fn itt_canon2(t1: [&Rat3; 3], t2: [&Rat3; 3], n1: &Rat3, n2: &Rat3) -> Itt {
    let [p1, q1, r1] = t1;
    let [p2, q2, r2] = t2;
    let p1p2 = sub3(p2, p1);
    // Top test in the classification tree.
    let overlap = if tti_above(p1, q1, r2, &p1p2) > 0 {
        if tti_above(p1, r1, r2, &p1p2) > 0 {
            // No overlap: [k l] [i j].
            None
        } else if tti_above(p1, r1, q2, &p1p2) > 0 {
            // Overlap is [k [i l] j]: i is on p1r1, l is on p2r2.
            Some((tti_interp(p1, r1, p2, n2), tti_interp(p2, r2, p1, n1)))
        } else {
            // Overlap is [i [k l] j]: k is on p2q2, l is on p2r2.
            Some((tti_interp(p2, q2, p1, n1), tti_interp(p2, r2, p1, n1)))
        }
    } else if tti_above(p1, q1, q2, &p1p2) < 0 {
        // No overlap: [i j] [k l].
        None
    } else if tti_above(p1, r1, q2, &p1p2) >= 0 {
        // Overlap is [k [i j] l]: i is on p1r1, j is on p1q1.
        Some((tti_interp(p1, r1, p2, n2), tti_interp(p1, q1, p2, n2)))
    } else {
        // Overlap is [i [k j] l]: k is on p2q2, j is on p1q1.
        Some((tti_interp(p2, q2, p1, n1), tti_interp(p1, q1, p2, n2)))
    };
    let Some((intersect_1, intersect_2)) = overlap else {
        return Itt::none();
    };
    if intersect_1 == intersect_2 {
        Itt::point(intersect_1)
    } else {
        Itt::segment(intersect_1, intersect_2)
    }
}

/// Blender `itt_canon1`: picks the canonical vertex order of triangle 2 given the canonical
/// order of triangle 1 (`t1`) and the signs `s2` of `t2`'s vertices against plane 1.
fn itt_canon1(t1: [&Rat3; 3], t2: [&Rat3; 3], s2: [i8; 3], n1: &Rat3, n2: &Rat3) -> Itt {
    let [p1, q1, r1] = t1;
    let [p2, q2, r2] = t2;
    let [sp2, sq2, sr2] = s2;
    if sp2 > 0 {
        if sq2 > 0 {
            return itt_canon2([p1, r1, q1], [r2, p2, q2], n1, n2);
        }
        if sr2 > 0 {
            return itt_canon2([p1, r1, q1], [q2, r2, p2], n1, n2);
        }
        return itt_canon2([p1, q1, r1], [p2, q2, r2], n1, n2);
    }
    if sp2 < 0 {
        if sq2 < 0 {
            return itt_canon2([p1, q1, r1], [r2, p2, q2], n1, n2);
        }
        if sr2 < 0 {
            return itt_canon2([p1, q1, r1], [q2, r2, p2], n1, n2);
        }
        return itt_canon2([p1, r1, q1], [p2, q2, r2], n1, n2);
    }
    if sq2 < 0 {
        if sr2 >= 0 {
            return itt_canon2([p1, r1, q1], [q2, r2, p2], n1, n2);
        }
        return itt_canon2([p1, q1, r1], [p2, q2, r2], n1, n2);
    }
    if sq2 > 0 {
        if sr2 > 0 {
            return itt_canon2([p1, r1, q1], [p2, q2, r2], n1, n2);
        }
        return itt_canon2([p1, q1, r1], [q2, r2, p2], n1, n2);
    }
    if sr2 > 0 {
        return itt_canon2([p1, q1, r1], [r2, p2, q2], n1, n2);
    }
    if sr2 < 0 {
        return itt_canon2([p1, r1, q1], [r2, p2, q2], n1, n2);
    }
    Itt::coplanar()
}

/// Blender's dispatch in `intersect_tri_tri`: puts `t1` into canonical order (the vertex on the
/// positive side first, or the vertex on the plane first), permutes `t2` to match, and calls
/// [`itt_canon1`]. `s1` and `s2` are the plane-side signs of `t1` and `t2`, aligned with the
/// vertex arrays.
fn itt_dispatch(
    t1: [&Rat3; 3],
    t2: [&Rat3; 3],
    s1: [i8; 3],
    s2: [i8; 3],
    n1: &Rat3,
    n2: &Rat3,
) -> Itt {
    let [p1, q1, r1] = t1;
    let [p2, q2, r2] = t2;
    let [sp1, sq1, sr1] = s1;
    let [sp2, sq2, sr2] = s2;
    if sp1 > 0 {
        if sq1 > 0 {
            itt_canon1([r1, p1, q1], [p2, r2, q2], [sp2, sr2, sq2], n1, n2)
        } else if sr1 > 0 {
            itt_canon1([q1, r1, p1], [p2, r2, q2], [sp2, sr2, sq2], n1, n2)
        } else {
            itt_canon1([p1, q1, r1], [p2, q2, r2], [sp2, sq2, sr2], n1, n2)
        }
    } else if sp1 < 0 {
        if sq1 < 0 {
            itt_canon1([r1, p1, q1], [p2, q2, r2], [sp2, sq2, sr2], n1, n2)
        } else if sr1 < 0 {
            itt_canon1([q1, r1, p1], [p2, q2, r2], [sp2, sq2, sr2], n1, n2)
        } else {
            itt_canon1([p1, q1, r1], [p2, r2, q2], [sp2, sr2, sq2], n1, n2)
        }
    } else if sq1 < 0 {
        if sr1 >= 0 {
            itt_canon1([q1, r1, p1], [p2, r2, q2], [sp2, sr2, sq2], n1, n2)
        } else {
            itt_canon1([p1, q1, r1], [p2, q2, r2], [sp2, sq2, sr2], n1, n2)
        }
    } else if sq1 > 0 {
        if sr1 > 0 {
            itt_canon1([p1, q1, r1], [p2, r2, q2], [sp2, sr2, sq2], n1, n2)
        } else {
            itt_canon1([q1, r1, p1], [p2, q2, r2], [sp2, sq2, sr2], n1, n2)
        }
    } else if sr1 > 0 {
        itt_canon1([r1, p1, q1], [p2, q2, r2], [sp2, sq2, sr2], n1, n2)
    } else if sr1 < 0 {
        itt_canon1([r1, p1, q1], [p2, r2, q2], [sp2, sr2, sq2], n1, n2)
    } else {
        Itt::coplanar()
    }
}

/// Blender `intersect_tri_tri`: the exact intersection of triangles `t1` and `t2` of `arena`.
#[must_use]
pub(super) fn intersect_tri_tri(arena: &Arena, t1: &Tri, t2: &Tri) -> Itt {
    let [vp1, vq1, vr1] = t1.v.map(|id| arena.vert(id));
    let [vp2, vq2, vr2] = t2.v.map(|id| arena.vert(id));

    // Signs of t1's vertices against the plane of t2, with the double filter first.
    let n2 = tri_normal(&vp2.co, &vq2.co, &vr2.co);
    let d_n2 = normal_f(&n2);
    let mut sp1 = filter_plane_side(vp1.co_f, vr2.co_f, d_n2);
    let mut sq1 = filter_plane_side(vq1.co_f, vr2.co_f, d_n2);
    let mut sr1 = filter_plane_side(vr1.co_f, vr2.co_f, d_n2);
    if same_strict_side([sp1, sq1, sr1]) {
        return Itt::none();
    }

    // Repeat for signs of t2's vertices against the plane of t1.
    let n1 = tri_normal(&vp1.co, &vq1.co, &vr1.co);
    let d_n1 = normal_f(&n1);
    let mut sp2 = filter_plane_side(vp2.co_f, vr1.co_f, d_n1);
    let mut sq2 = filter_plane_side(vq2.co_f, vr1.co_f, d_n1);
    let mut sr2 = filter_plane_side(vr2.co_f, vr1.co_f, d_n1);
    if same_strict_side([sp2, sq2, sr2]) {
        return Itt::none();
    }

    let (p1, q1, r1) = (&vp1.co, &vq1.co, &vr1.co);
    let (p2, q2, r2) = (&vp2.co, &vq2.co, &vr2.co);

    // Undecided (zero) signs are decided exactly.
    if sp1 == 0 {
        sp1 = plane_side_exact(p1, r2, &n2);
    }
    if sq1 == 0 {
        sq1 = plane_side_exact(q1, r2, &n2);
    }
    if sr1 == 0 {
        sr1 = plane_side_exact(r1, r2, &n2);
    }
    if same_strict_side([sp1, sq1, sr1]) {
        return Itt::none();
    }

    if sp2 == 0 {
        sp2 = plane_side_exact(p2, r1, &n1);
    }
    if sq2 == 0 {
        sq2 = plane_side_exact(q2, r1, &n1);
    }
    if sr2 == 0 {
        sr2 = plane_side_exact(r2, r1, &n1);
    }
    if same_strict_side([sp2, sq2, sr2]) {
        return Itt::none();
    }

    itt_dispatch(
        [p1, q1, r1],
        [p2, q2, r2],
        [sp1, sq1, sr1],
        [sp2, sq2, sr2],
        &n1,
        &n2,
    )
}

#[cfg(test)]
mod tests {
    use super::super::super::imesh::NO_INDEX;
    use super::*;

    fn int(n: i64) -> Rat {
        Rat::from_integer(n.into())
    }

    fn frac(n: i64, d: i64) -> Rat {
        Rat::new(n.into(), d.into())
    }

    fn pt(x: i64, y: i64, z: i64) -> Rat3 {
        [int(x), int(y), int(z)]
    }

    fn make_tri(arena: &mut Arena, a: Rat3, b: Rat3, c: Rat3) -> Tri {
        let v = [
            arena.add_or_find_vert(a, NO_INDEX),
            arena.add_or_find_vert(b, NO_INDEX),
            arena.add_or_find_vert(c, NO_INDEX),
        ];
        Tri::new(v, 0, [NO_INDEX; 3])
    }

    /// Transversal crossing. T1 = (0,0,0),(4,0,0),(0,4,0) lies in z = 0. T2 lies in x = 1 and
    /// meets z = 0 along (1,-1,0)..(1,2,0). Their common part is (1,0,0)..(1,2,0), limited by
    /// T1's edge y = 0 at (1,0,0) and T2's edge EF at (1,2,0).
    #[test]
    fn transversal_crossing_is_segment() {
        let mut arena = Arena::new();
        let t1 = make_tri(&mut arena, pt(0, 0, 0), pt(4, 0, 0), pt(0, 4, 0));
        let t2 = make_tri(&mut arena, pt(1, -1, -1), pt(1, 5, -1), pt(1, -1, 1));
        let itt = intersect_tri_tri(&arena, &t1, &t2);
        assert_eq!(itt.kind, IttKind::Segment);
        assert_eq!(itt.p1, pt(1, 0, 0));
        assert_eq!(itt.p2, pt(1, 2, 0));
    }

    /// Touching at one vertex. T2 = (0,0,0),(1,0,1),(0,1,1) lies in z >= 0 and meets z = 0 only
    /// at the shared vertex (0,0,0), which is a vertex of T1.
    #[test]
    fn touching_at_vertex_is_point() {
        let mut arena = Arena::new();
        let t1 = make_tri(&mut arena, pt(0, 0, 0), pt(4, 0, 0), pt(0, 4, 0));
        let t2 = make_tri(&mut arena, pt(0, 0, 0), pt(1, 0, 1), pt(0, 1, 1));
        let itt = intersect_tri_tri(&arena, &t1, &t2);
        assert_eq!(itt.kind, IttKind::Point);
        assert_eq!(itt.p1, pt(0, 0, 0));
    }

    #[test]
    fn far_apart_parallel_planes_are_none() {
        let mut arena = Arena::new();
        let t1 = make_tri(&mut arena, pt(0, 0, 0), pt(4, 0, 0), pt(0, 4, 0));
        let t2 = make_tri(&mut arena, pt(10, 10, 1), pt(11, 10, 1), pt(10, 11, 1));
        assert_eq!(intersect_tri_tri(&arena, &t1, &t2).kind, IttKind::None);
    }

    /// Coplanar overlapping pair: both in z = 0. The second triangle lies inside the first and
    /// touches its hypotenuse x + y = 4 at (3,1,0) and (1,3,0).
    #[test]
    fn coplanar_overlap_is_coplanar() {
        let mut arena = Arena::new();
        let t1 = make_tri(&mut arena, pt(0, 0, 0), pt(4, 0, 0), pt(0, 4, 0));
        let t2 = make_tri(&mut arena, pt(1, 1, 0), pt(3, 1, 0), pt(1, 3, 0));
        assert_eq!(intersect_tri_tri(&arena, &t1, &t2).kind, IttKind::Coplanar);
    }

    /// Coplanar disjoint pair. Blender reports every coplanar pair as `Coplanar`; the overlap
    /// test belongs to the coplanar-cluster stage, so this is `Coplanar`, not `None`.
    #[test]
    fn coplanar_disjoint_is_still_coplanar() {
        let mut arena = Arena::new();
        let t1 = make_tri(&mut arena, pt(0, 0, 0), pt(4, 0, 0), pt(0, 4, 0));
        let t2 = make_tri(&mut arena, pt(5, 5, 0), pt(6, 5, 0), pt(5, 6, 0));
        assert_eq!(intersect_tri_tri(&arena, &t1, &t2).kind, IttKind::Coplanar);
    }

    /// Edge through the other's interior. T2's edge PQ = (1,1,-1)..(1,1,1) pierces T1 at
    /// (1,1,0). T2 meets z = 0 along (1,1,0)..(3,-1,0). T1 cuts that segment at its edge
    /// y = 0, at (2,0,0), so the common part is (2,0,0)..(1,1,0).
    #[test]
    fn edge_through_interior_is_segment() {
        let mut arena = Arena::new();
        let t1 = make_tri(&mut arena, pt(0, 0, 0), pt(4, 0, 0), pt(0, 4, 0));
        let t2 = make_tri(&mut arena, pt(1, 1, -1), pt(1, 1, 1), pt(3, -1, 0));
        let itt = intersect_tri_tri(&arena, &t1, &t2);
        assert_eq!(itt.kind, IttKind::Segment);
        assert_eq!(itt.p1, pt(2, 0, 0));
        assert_eq!(itt.p2, pt(1, 1, 0));
    }

    /// Vertex on the other's plane. S = (5,0,0) lies on z = 0 outside T1. T2 meets z = 0 along
    /// (3,1,0)..(5,0,0), and T1 contains only its endpoint (3,1,0) on edge BC.
    #[test]
    fn vertex_on_plane_gives_point() {
        let mut arena = Arena::new();
        let t1 = make_tri(&mut arena, pt(0, 0, 0), pt(4, 0, 0), pt(0, 4, 0));
        let t2 = make_tri(&mut arena, pt(5, 0, 0), pt(3, 1, -1), pt(3, 1, 1));
        let itt = intersect_tri_tri(&arena, &t1, &t2);
        assert_eq!(itt.kind, IttKind::Point);
        assert_eq!(itt.p1, pt(3, 1, 0));
    }

    /// Non-integer endpoint. T1 = (0,0,0),(3,0,0),(0,3,0) in z = 0; T2 = (1,-1,-1),(1,1,-1),
    /// (1,-1,2) in x = 1. T2 meets z = 0 along (1,-1,0)..(1,1/3,0). Inside T1 the part is
    /// (1,0,0)..(1,1/3,0).
    #[test]
    fn rational_endpoint_is_exact() {
        let mut arena = Arena::new();
        let t1 = make_tri(&mut arena, pt(0, 0, 0), pt(3, 0, 0), pt(0, 3, 0));
        let t2 = make_tri(&mut arena, pt(1, -1, -1), pt(1, 1, -1), pt(1, -1, 2));
        let itt = intersect_tri_tri(&arena, &t1, &t2);
        assert_eq!(itt.kind, IttKind::Segment);
        assert_eq!(itt.p1, pt(1, 0, 0));
        assert_eq!(itt.p2, [int(1), frac(1, 3), int(0)]);
    }
}
