//! Exact constrained Delaunay triangulation for the Mesh Arrangements port (PR17a-1).
//!
//! This is Blender's `delaunay_2d_calc<mpq_class>(input, CDT_INSIDE)` (`blenlib/intern/delaunay_2d.cc`,
//! Blender 5.2). `mesh_intersect` calls it to triangulate each input triangle once its intersection
//! segments are known, and to label every output triangle with the input faces it lies in.
//!
//! # Ported
//! - Input vertices are deduplicated by exact position (the lowest input index survives), then
//!   triangulated with the Guibas–Stolfi divide-and-conquer Delaunay algorithm.
//! - Edge and face constraints are inserted one at a time by walking the arrangement. Crossed
//!   constrained edges are split at the exact intersection point, crossed unconstrained edges are
//!   deleted, and the resulting pseudo-polygons are re-triangulated (Anglada's method) so the result
//!   stays constrained Delaunay.
//! - Face labels: every constrained input edge carries an encoded id (`face_edge_offset`). Output
//!   triangles inside an input face receive that face's index by a flood fill that does not cross
//!   the face's boundary.
//! - `CDT_INSIDE` output: every face reachable from the outer face without crossing a constraint
//!   is removed.
//! - Output `vert_orig`, `edge_orig`, and `face_orig` follow Blender's `CDT_result`.
//!
//! Orientation and incircle tests use Blender's double-precision error-bounded filters and fall back
//! to exact rational arithmetic when the filter is undecided. Intersection parameters are exact.
//!
//! # Not ported
//! - The `double` instantiation and `epsilon` snapping: exact arithmetic never uses epsilon, and the
//!   boolean path passes `epsilon = 0`.
//! - Output types other than `CDT_INSIDE`: `CDT_FULL`, `CDT_CONSTRAINTS`,
//!   `CDT_CONSTRAINTS_VALID_BMESH*`, and every `*_WITH_HOLES*` variant with its even-odd and
//!   non-zero hole detectors and their winding and polygon-boundary maps.
//! - `need_ids == false`: ids are always produced, because `face_orig` is part of the contract.
//! - Debug dumps and drawing.
//!
//! # Deviations
//! - Blender's `Set<uint32_t>` iteration order is unspecified. Id sets are `BTreeSet`s, so every
//!   `*_orig` list is ascending.
//! - Blender's `CDT_INSIDE` faces are triangles, and its consumer asserts so. `CdtOutput::faces`
//!   is `[u32; 3]`, so a polygon with more than three sides is fan-triangulated behind a debug
//!   assertion. Polygons with fewer than three sides are dropped.
//! - `CdtOutput::face_edge_offset` is added beyond the contract. Consumers decode `edge_orig`
//!   with it, exactly as Blender's `CDT_result::face_edge_offset`.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use num_traits::{One, Signed, Zero};

use super::exact::{Rat, Rat2, orient2d as exact_orient2d, rat_to_f64, sgn};

/// Index into an arrangement array (vertex, symedge, edge, or face).
type Id = u32;

/// Null index (Blender `nullptr`).
const NIL: Id = u32::MAX;

/// Error-bound index of Blender's `filtered_orient2d`.
const INDEX_ORIENT2D: f64 = 6.0;
/// Error-bound index of Blender's `in_line`.
const INDEX_IN_LINE: f64 = 6.0;
/// Error-bound index of Blender's `filtered_incircle`.
const INDEX_INCIRCLE: f64 = 14.0;
/// Blender aborts a crossing walk that grows past this many crossings.
const UNREASONABLY_LARGE_CROSSINGS: usize = 100_000;

/// Input to the triangulation (Blender `CDT_input<mpq_class>`).
pub(crate) struct CdtInput {
    /// Exact vertex positions. Coincident vertices are merged.
    pub verts: Vec<Rat2>,
    /// Constraint edges as pairs of vertex indices. Pairs with an out-of-range index are ignored.
    pub edges: Vec<(u32, u32)>,
    /// Input polygons as counter-clockwise loops of vertex indices. Their interiors are kept.
    pub faces: Vec<Vec<u32>>,
}

/// Output of the triangulation (Blender `CDT_result<mpq_class>` for `CDT_INSIDE`).
#[derive(Debug, Default)]
pub(crate) struct CdtOutput {
    /// Output vertex positions. Coincident inputs are merged, and created points are appended.
    pub verts: Vec<Rat2>,
    /// Triangles as counter-clockwise loops of output vertex indices.
    pub faces: Vec<[u32; 3]>,
    /// For each output triangle, the input face indices it lies in.
    pub face_orig: Vec<Vec<u32>>,
    /// Output edges as pairs of output vertex indices.
    pub edges: Vec<(u32, u32)>,
    /// For each output edge, the ids it overlaps. An id below `face_edge_offset` indexes
    /// `CdtInput::edges`. Otherwise `id / face_edge_offset - 1` is the input face and
    /// `id % face_edge_offset` is the position within that face.
    pub edge_orig: Vec<Vec<u32>>,
    /// For each output vertex, the input vertex indices it stands for. Empty for created vertices.
    pub vert_orig: Vec<Vec<u32>>,
    /// Base of the `edge_orig` encoding, as in Blender's `CDT_result::face_edge_offset`.
    pub face_edge_offset: u32,
}

/// Blender `FatCo<mpq_class>`: an exact coordinate with double approximations for the filters.
#[derive(Clone, Debug)]
struct FatCo {
    exact: Rat2,
    approx: [f64; 2],
    abs_approx: [f64; 2],
}

impl FatCo {
    fn new(exact: Rat2) -> Self {
        let approx = [rat_to_f64(&exact[0]), rat_to_f64(&exact[1])];
        let abs_approx = [approx[0].abs(), approx[1].abs()];
        Self {
            exact,
            approx,
            abs_approx,
        }
    }
}

/// Blender `SymEdge`: one directed half of an undirected edge.
#[derive(Clone, Copy, Debug)]
struct SymEdge {
    /// Next symedge around the face on this symedge's left, CCW.
    next: Id,
    /// Next symedge counter-clockwise around the origin vertex.
    rot: Id,
    /// Origin vertex.
    vert: Id,
    /// Face on the left.
    face: Id,
}

/// Blender `CDTVert`.
#[derive(Debug)]
struct CdtVert {
    co: FatCo,
    /// Some symedge leaving this vertex, or `NIL` when isolated.
    symedge: Id,
    /// Input vertex ids merged into this vertex.
    input_ids: BTreeSet<u32>,
    /// Vertex this one merged into, or `NIL`.
    merge_to: Id,
    /// Visit epoch used while walking crossings.
    visit_index: u32,
}

/// Blender `CDTFace`.
#[derive(Debug)]
struct CdtFace {
    /// A symedge with this face on its left. Only valid during output.
    symedge: Id,
    /// Input face ids that cover this face.
    input_ids: BTreeSet<u32>,
    /// Visit epoch used by flood fills.
    visit_index: u32,
    /// Set once the face no longer exists.
    deleted: bool,
}

impl CdtFace {
    fn new() -> Self {
        Self {
            symedge: NIL,
            input_ids: BTreeSet::new(),
            visit_index: 0,
            deleted: false,
        }
    }
}

/// Blender `CrossData`: one point along the path of a constraint being inserted.
#[derive(Clone, Debug)]
struct CrossData {
    /// Fraction along `in_`, or `0` when this crossing is a vertex. A negative value marks a
    /// crossing that was discarded.
    lambda: Rat,
    /// Vertex crossing, or the created split vertex once an edge crossing has been split.
    vert: Id,
    /// Symedge crossed (edge crossing), or the attachment symedge (vertex crossing).
    in_: Id,
    /// Symedge leaving this crossing toward the next one.
    out: Id,
}

impl CrossData {
    fn empty() -> Self {
        Self {
            lambda: Rat::zero(),
            vert: NIL,
            in_: NIL,
            out: NIL,
        }
    }
}

/// Outcome of a segment-segment intersection test (Blender `isect_result`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IsectKind {
    /// Parallel lines (Blender `LINE_LINE_COLINEAR`).
    Colinear,
    /// Segments do not meet (Blender `LINE_LINE_NONE`).
    Miss,
    /// Segments meet at an endpoint of one of them (Blender `LINE_LINE_EXACT`).
    Exact,
    /// Segments cross strictly inside both (Blender `LINE_LINE_CROSS`).
    Cross,
}

/// Blender's `ISECT_LINE_LINE_*` test for exact segments. `lambda` is the parameter along `v1 -> v2`.
fn isect_seg_seg(v1: &Rat2, v2: &Rat2, v3: &Rat2, v4: &Rat2) -> (IsectKind, Rat) {
    let d1 = [&v2[0] - &v1[0], &v2[1] - &v1[1]];
    let d2 = [&v4[0] - &v3[0], &v4[1] - &v3[1]];
    let div = &d1[0] * &d2[1] - &d1[1] * &d2[0];
    if div.is_zero() {
        return (IsectKind::Colinear, Rat::zero());
    }
    let w = [&v1[0] - &v3[0], &v1[1] - &v3[1]];
    let lambda = (&w[1] * &d2[0] - &w[0] * &d2[1]) / &div;
    let mu = (&w[1] * &d1[0] - &w[0] * &d1[1]) / &div;
    let unit = Rat::zero()..=Rat::one();
    if unit.contains(&lambda) && unit.contains(&mu) {
        let endpoint = lambda.is_zero() || lambda == Rat::one() || mu.is_zero() || mu == Rat::one();
        let kind = if endpoint {
            IsectKind::Exact
        } else {
            IsectKind::Cross
        };
        (kind, lambda)
    } else {
        (IsectKind::Miss, lambda)
    }
}

/// Exact point `a + t (b - a)`, as Blender's `interpolate`.
fn interpolate(a: &Rat2, b: &Rat2, t: &Rat) -> Rat2 {
    let one_minus_t = Rat::one() - t;
    [
        &a[0] * &one_minus_t + &b[0] * t,
        &a[1] * &one_minus_t + &b[1] * t,
    ]
}

fn sub2(a: &Rat2, b: &Rat2) -> Rat2 {
    [&a[0] - &b[0], &a[1] - &b[1]]
}

fn dot2(a: &Rat2, b: &Rat2) -> Rat {
    &a[0] * &b[0] + &a[1] * &b[1]
}

/// Exact `incircle`: positive when `d` lies inside the circle through CCW `a, b, c`.
fn exact_incircle(a: &Rat2, b: &Rat2, c: &Rat2, d: &Rat2) -> i8 {
    let adx = &a[0] - &d[0];
    let bdx = &b[0] - &d[0];
    let cdx = &c[0] - &d[0];
    let ady = &a[1] - &d[1];
    let bdy = &b[1] - &d[1];
    let cdy = &c[1] - &d[1];
    let alift = &adx * &adx + &ady * &ady;
    let blift = &bdx * &bdx + &bdy * &bdy;
    let clift = &cdx * &cdx + &cdy * &cdy;
    let det = &alift * (&bdx * &cdy - &cdx * &bdy)
        + &blift * (&cdx * &ady - &adx * &cdy)
        + &clift * (&adx * &bdy - &bdx * &ady);
    sgn(&det)
}

/// Blender `filtered_orient2d`: double filter, then exact orientation of `a, b, c`.
fn filtered_orient2d(a: &FatCo, b: &FatCo, c: &FatCo) -> i8 {
    let det = (a.approx[0] - c.approx[0]) * (b.approx[1] - c.approx[1])
        - (a.approx[1] - c.approx[1]) * (b.approx[0] - c.approx[0]);
    let supremum = (a.abs_approx[0] + c.abs_approx[0]) * (b.abs_approx[1] + c.abs_approx[1])
        + (a.abs_approx[1] + c.abs_approx[1]) * (b.abs_approx[0] + c.abs_approx[0]);
    let err_bound = supremum * INDEX_ORIENT2D * f64::EPSILON;
    if det.abs() > err_bound {
        return if det > 0.0 { 1 } else { -1 };
    }
    exact_orient2d(&a.exact, &b.exact, &c.exact)
}

/// Blender `filtered_incircle`: double filter, then exact incircle test.
fn filtered_incircle(a: &FatCo, b: &FatCo, c: &FatCo, d: &FatCo) -> i8 {
    let adx = a.approx[0] - d.approx[0];
    let bdx = b.approx[0] - d.approx[0];
    let cdx = c.approx[0] - d.approx[0];
    let ady = a.approx[1] - d.approx[1];
    let bdy = b.approx[1] - d.approx[1];
    let cdy = c.approx[1] - d.approx[1];
    let bdxcdy = bdx * cdy;
    let cdxbdy = cdx * bdy;
    let alift = adx * adx + ady * ady;
    let cdxady = cdx * ady;
    let adxcdy = adx * cdy;
    let blift = bdx * bdx + bdy * bdy;
    let adxbdy = adx * bdy;
    let bdxady = bdx * ady;
    let clift = cdx * cdx + cdy * cdy;
    let det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady);

    let sup_adx = a.abs_approx[0] + d.abs_approx[0];
    let sup_bdx = b.abs_approx[0] + d.abs_approx[0];
    let sup_cdx = c.abs_approx[0] + d.abs_approx[0];
    let sup_ady = a.abs_approx[1] + d.abs_approx[1];
    let sup_bdy = b.abs_approx[1] + d.abs_approx[1];
    let sup_cdy = c.abs_approx[1] + d.abs_approx[1];
    let sup_bdxcdy = sup_bdx * sup_cdy;
    let sup_cdxbdy = sup_cdx * sup_bdy;
    let sup_alift = sup_adx * sup_adx + sup_ady * sup_ady;
    let sup_cdxady = sup_cdx * sup_ady;
    let sup_adxcdy = sup_adx * sup_cdy;
    let sup_blift = sup_bdx * sup_bdx + sup_bdy * sup_bdy;
    let sup_adxbdy = sup_adx * sup_bdy;
    let sup_bdxady = sup_bdx * sup_ady;
    let sup_clift = sup_cdx * sup_cdx + sup_cdy * sup_cdy;
    let sup_det = sup_alift * (sup_bdxcdy + sup_cdxbdy)
        + sup_blift * (sup_cdxady + sup_adxcdy)
        + sup_clift * (sup_adxbdy + sup_bdxady);
    let err_bound = sup_det * INDEX_INCIRCLE * f64::EPSILON;
    if det.abs() > err_bound {
        return if det < 0.0 { -1 } else { i8::from(det > 0.0) };
    }
    exact_incircle(&a.exact, &b.exact, &c.exact, &d.exact)
}

/// Blender `in_line`: whether `a -- b -- c` are in that order, given they are collinear.
fn in_line(a: &FatCo, b: &FatCo, c: &FatCo) -> bool {
    let ab = [b.approx[0] - a.approx[0], b.approx[1] - a.approx[1]];
    let bc = [c.approx[0] - b.approx[0], c.approx[1] - b.approx[1]];
    let ac = [c.approx[0] - a.approx[0], c.approx[1] - a.approx[1]];
    let sup_ab = [
        b.abs_approx[0] + a.abs_approx[0],
        b.abs_approx[1] + a.abs_approx[1],
    ];
    let sup_bc = [
        c.abs_approx[0] + b.abs_approx[0],
        c.abs_approx[1] + b.abs_approx[1],
    ];
    let sup_ac = [
        c.abs_approx[0] + a.abs_approx[0],
        c.abs_approx[1] + a.abs_approx[1],
    ];
    let dot_ab_ac = ab[0] * ac[0] + ab[1] * ac[1];
    let sup_dot_ab_ac = sup_ab[0] * sup_ac[0] + sup_ab[1] * sup_ac[1];
    if dot_ab_ac < -(sup_dot_ab_ac * INDEX_IN_LINE * f64::EPSILON) {
        return false;
    }
    let dot_bc_ac = bc[0] * ac[0] + bc[1] * ac[1];
    let sup_dot_bc_ac = sup_bc[0] * sup_ac[0] + sup_bc[1] * sup_ac[1];
    if dot_bc_ac < -(sup_dot_bc_ac * INDEX_IN_LINE * f64::EPSILON) {
        return false;
    }
    let exact_ab = sub2(&b.exact, &a.exact);
    let exact_ac = sub2(&c.exact, &a.exact);
    if dot2(&exact_ab, &exact_ac) < Rat::zero() {
        return false;
    }
    let exact_bc = sub2(&c.exact, &b.exact);
    dot2(&exact_bc, &exact_ac) >= Rat::zero()
}

/// Squared distance between double approximations, as Blender's `distance_squared` on `approx`.
fn approx_dist_sq(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn to_id(n: usize) -> Id {
    n as Id
}

fn to_u32(n: usize) -> u32 {
    n as u32
}

fn ix(id: Id) -> usize {
    id as usize
}

/// First symedge of undirected edge `e`.
fn s0(e: Id) -> Id {
    e * 2
}

/// Undirected edge of symedge `se`.
fn edge_of(se: Id) -> Id {
    se / 2
}

/// Blender `CDT_state<mpq_class>` together with its `CDTArrangement`.
///
/// Symedges of edge `e` are `2e` and `2e + 1`, which replace Blender's pointer pairs. Deleted
/// edges keep their slots with `next == NIL`, so every id stays valid for the whole run.
struct Cdt {
    verts: Vec<CdtVert>,
    symedges: Vec<SymEdge>,
    edge_ids: Vec<BTreeSet<u32>>,
    faces: Vec<CdtFace>,
    outer: Id,
    input_vert_num: usize,
    visit_count: u32,
    face_edge_offset: u32,
}

impl Cdt {
    fn new(input_vert_num: usize) -> Self {
        Self {
            verts: Vec::with_capacity(2 * input_vert_num),
            symedges: Vec::new(),
            edge_ids: Vec::new(),
            faces: vec![CdtFace::new()],
            outer: 0,
            input_vert_num,
            visit_count: 0,
            face_edge_offset: 0,
        }
    }

    fn sym_edge(&self, se: Id) -> &SymEdge {
        &self.symedges[ix(se)]
    }

    fn sym_edge_mut(&mut self, se: Id) -> &mut SymEdge {
        &mut self.symedges[ix(se)]
    }

    fn next(&self, se: Id) -> Id {
        self.sym_edge(se).next
    }

    fn rot(&self, se: Id) -> Id {
        self.sym_edge(se).rot
    }

    fn vert(&self, se: Id) -> Id {
        self.sym_edge(se).vert
    }

    fn face(&self, se: Id) -> Id {
        self.sym_edge(se).face
    }

    /// Blender `sym`: the other symedge of the same edge, `se->next->rot`.
    fn sym(&self, se: Id) -> Id {
        self.rot(self.next(se))
    }

    /// Blender `prev`: the symedge whose `next` is `se`, `se->rot->next->rot`.
    fn prev(&self, se: Id) -> Id {
        self.rot(self.next(self.rot(se)))
    }

    fn co(&self, v: Id) -> &FatCo {
        &self.verts[ix(v)].co
    }

    fn is_constrained(&self, se: Id) -> bool {
        !self.edge_ids[ix(edge_of(se))].is_empty()
    }

    fn is_deleted_edge(&self, e: Id) -> bool {
        self.sym_edge(s0(e)).next == NIL
    }

    fn orient_v(&self, a: Id, b: Id, c: Id) -> i8 {
        filtered_orient2d(self.co(a), self.co(b), self.co(c))
    }

    fn incircle_v(&self, a: Id, b: Id, c: Id, d: Id) -> i8 {
        filtered_incircle(self.co(a), self.co(b), self.co(c), self.co(d))
    }

    /// Blender `CDTArrangement::add_vert`.
    fn add_vert(&mut self, co: Rat2) -> Id {
        let id = to_id(self.verts.len());
        self.verts.push(CdtVert {
            co: FatCo::new(co),
            symedge: NIL,
            input_ids: BTreeSet::new(),
            merge_to: NIL,
            visit_index: 0,
        });
        id
    }

    /// Blender `CDTArrangement::add_face`.
    fn add_face(&mut self) -> Id {
        let id = to_id(self.faces.len());
        self.faces.push(CdtFace::new());
        id
    }

    /// Blender `CDTArrangement::add_edge`. Returns the new undirected edge; its symedges are
    /// `s0(e)` (from `v1`, left face `fleft`) and `s0(e) + 1` (from `v2`, left face `fright`).
    fn add_edge(&mut self, v1: Id, v2: Id, fleft: Id, fright: Id) -> Id {
        let e = to_id(self.edge_ids.len());
        let s = s0(e);
        self.edge_ids.push(BTreeSet::new());
        self.symedges.push(SymEdge {
            next: NIL,
            rot: NIL,
            vert: v1,
            face: fleft,
        });
        self.symedges.push(SymEdge {
            next: NIL,
            rot: NIL,
            vert: v2,
            face: fright,
        });
        let first = &mut self.verts[ix(v1)];
        if first.symedge == NIL {
            first.symedge = s;
        }
        let second = &mut self.verts[ix(v2)];
        if second.symedge == NIL {
            second.symedge = s + 1;
        }
        e
    }

    /// Blender `add_vert_to_symedge_edge`: connects new vertex `v` to the origin of `se`'s
    /// destination side, splicing it into the rotation at `se`.
    fn add_vert_to_symedge_edge(&mut self, v: Id, se: Id) -> Id {
        let se_rot = self.rot(se);
        let se_rotsym = self.sym(se_rot);
        let se_vert = self.vert(se);
        let se_face = self.face(se);
        let e = self.add_edge(v, se_vert, se_face, se_face);
        let new_se = s0(e);
        let new_se_sym = new_se + 1;
        self.sym_edge_mut(new_se).next = se;
        self.sym_edge_mut(new_se_sym).next = new_se;
        self.sym_edge_mut(new_se).rot = new_se;
        self.sym_edge_mut(new_se_sym).rot = se_rot;
        self.sym_edge_mut(se).rot = new_se_sym;
        self.sym_edge_mut(se_rotsym).next = new_se_sym;
        e
    }

    /// Blender `connect_separate_parts`: joins two outer-boundary symedges of disjoint parts.
    fn connect_separate_parts(&mut self, se1: Id, se2: Id) -> Id {
        let se1_rot = self.rot(se1);
        let se1_rotsym = self.sym(se1_rot);
        let se2_rot = self.rot(se2);
        let se2_rotsym = self.sym(se2_rot);
        let outer = self.outer;
        let v1 = self.vert(se1);
        let v2 = self.vert(se2);
        let e = self.add_edge(v1, v2, outer, outer);
        let new_se = s0(e);
        let new_se_sym = new_se + 1;
        self.sym_edge_mut(new_se).next = se2;
        self.sym_edge_mut(new_se_sym).next = se1;
        self.sym_edge_mut(new_se).rot = se1_rot;
        self.sym_edge_mut(new_se_sym).rot = se2_rot;
        self.sym_edge_mut(se1).rot = new_se;
        self.sym_edge_mut(se2).rot = new_se_sym;
        self.sym_edge_mut(se1_rotsym).next = new_se;
        self.sym_edge_mut(se2_rotsym).next = new_se_sym;
        e
    }

    /// Blender `add_diagonal`: splits a face of more than three sides with a new edge from
    /// `s1`'s origin to `s2`'s origin. `s1`'s face keeps the `s1` side, and a new face takes `s2`.
    fn add_diagonal(&mut self, s1: Id, s2: Id) -> Id {
        let fold = self.face(s1);
        let fnew = self.add_face();
        let s1prev = self.prev(s1);
        let s1prevsym = self.sym(s1prev);
        let s2prev = self.prev(s2);
        let s2prevsym = self.sym(s2prev);
        let v1 = self.vert(s1);
        let v2 = self.vert(s2);
        let e = self.add_edge(v1, v2, fnew, fold);
        let sdiag = s0(e);
        let sdiagsym = sdiag + 1;
        self.sym_edge_mut(sdiag).next = s2;
        self.sym_edge_mut(sdiagsym).next = s1;
        self.sym_edge_mut(s2prev).next = sdiagsym;
        self.sym_edge_mut(s1prev).next = sdiag;
        self.sym_edge_mut(s1).rot = sdiag;
        self.sym_edge_mut(sdiag).rot = s1prevsym;
        self.sym_edge_mut(s2).rot = sdiagsym;
        self.sym_edge_mut(sdiagsym).rot = s2prevsym;
        let mut se = s2;
        while se != sdiag {
            self.sym_edge_mut(se).face = fnew;
            se = self.next(se);
        }
        let inherited = self.faces[ix(fold)].input_ids.clone();
        self.faces[ix(fnew)].input_ids.extend(inherited);
        e
    }

    /// Blender `split_edge`: splits the edge of `se` at fraction `lambda`. Returns the new second half.
    fn split_edge(&mut self, se: Id, lambda: &Rat) -> Id {
        let point = interpolate(
            &self.co(self.vert(se)).exact,
            &self.co(self.vert(self.next(se))).exact,
            lambda,
        );
        let sesym = self.sym(se);
        let sesymprev = self.prev(sesym);
        let sesymprevsym = self.sym(sesymprev);
        let senext = self.next(se);
        let dest = self.vert(self.next(se));
        let v = self.add_vert(point);
        let se_face = self.face(se);
        let sesym_face = self.face(sesym);
        let e = self.add_edge(v, dest, se_face, sesym_face);
        self.sym_edge_mut(sesym).vert = v;
        let newse = s0(e);
        let newsesym = newse + 1;
        self.sym_edge_mut(se).next = newse;
        self.sym_edge_mut(newsesym).next = sesym;
        self.sym_edge_mut(newse).next = senext;
        self.sym_edge_mut(newse).rot = sesym;
        self.sym_edge_mut(sesym).rot = newse;
        self.sym_edge_mut(senext).rot = newsesym;
        self.sym_edge_mut(newsesym).rot = sesymprevsym;
        self.sym_edge_mut(sesymprev).next = newsesym;
        let new_origin = self.vert(newsesym);
        if self.verts[ix(new_origin)].symedge == sesym {
            self.verts[ix(new_origin)].symedge = newsesym;
        }
        let inherited = self.edge_ids[ix(edge_of(se))].clone();
        self.edge_ids[ix(e)] = inherited;
        e
    }

    /// Blender `CDTArrangement::delete_edge`. The face on the other side is merged into `aface`
    /// and marked deleted, and the outer face is kept whenever it is one of the two sides.
    fn delete_edge(&mut self, se: Id) {
        let sesym = self.sym(se);
        let v1 = self.vert(se);
        let v2 = self.vert(sesym);
        let aface = self.face(se);
        let bface = self.face(sesym);
        let f = self.next(se);
        let h = self.prev(se);
        let i = self.next(sesym);
        let j = self.prev(sesym);
        let jsym = self.sym(j);
        let hsym = self.sym(h);
        let v1_isolated = i == se;
        let v2_isolated = f == sesym;

        if !v1_isolated {
            self.sym_edge_mut(h).next = i;
            self.sym_edge_mut(i).rot = hsym;
        }
        if !v2_isolated {
            self.sym_edge_mut(j).next = f;
            self.sym_edge_mut(f).rot = jsym;
        }
        if !v1_isolated && !v2_isolated && aface != bface {
            let mut k = i;
            while k != f {
                self.sym_edge_mut(k).face = aface;
                k = self.next(k);
            }
        }

        if v1_isolated {
            self.verts[ix(v1)].symedge = NIL;
        } else if self.verts[ix(v1)].symedge == se {
            self.verts[ix(v1)].symedge = i;
        }
        if v2_isolated {
            self.verts[ix(v2)].symedge = NIL;
        } else if self.verts[ix(v2)].symedge == sesym {
            self.verts[ix(v2)].symedge = f;
        }

        for half in [se, sesym] {
            let sed = self.sym_edge_mut(half);
            sed.next = NIL;
            sed.rot = NIL;
        }

        if !v1_isolated && !v2_isolated && aface != bface {
            self.faces[ix(bface)].deleted = true;
            if self.outer == bface {
                self.outer = aface;
            }
        } else if v1_isolated && v2_isolated {
            let outer = self.outer;
            if aface == outer {
                let outer_se = self.faces[ix(outer)].symedge;
                if outer_se == se || outer_se == sesym {
                    self.faces[ix(outer)].symedge = NIL;
                }
            } else {
                self.faces[ix(aface)].deleted = true;
            }
        }
    }

    /// Blender `find_symedge_between_verts`: the symedge from `v1` to `v2`, or `NIL`.
    fn find_symedge_between_verts(&self, v1: Id, v2: Id) -> Id {
        let start = self.verts[ix(v1)].symedge;
        if start == NIL {
            return NIL;
        }
        let mut t = start;
        loop {
            if self.vert(self.next(t)) == v2 {
                return t;
            }
            t = self.rot(t);
            if t == start {
                return NIL;
            }
        }
    }

    /// Blender `find_symedge_with_face`: a symedge leaving `v` with face `f` on its left, or `NIL`.
    fn find_symedge_with_face(&self, v: Id, f: Id) -> Id {
        let start = self.verts[ix(v)].symedge;
        if start == NIL {
            return NIL;
        }
        let mut t = start;
        loop {
            if self.face(t) == f {
                return t;
            }
            t = self.rot(t);
            if t == start {
                return NIL;
            }
        }
    }

    fn exists_edge(&self, v1: Id, v2: Id) -> bool {
        self.find_symedge_between_verts(v1, v2) != NIL
    }

    /// Blender `vert_left_of_symedge`.
    fn vert_left_of_symedge(&self, v: Id, se: Id) -> bool {
        self.orient_v(v, self.vert(se), self.vert(self.next(se))) > 0
    }

    /// Blender `vert_right_of_symedge`.
    fn vert_right_of_symedge(&self, v: Id, se: Id) -> bool {
        self.orient_v(v, self.vert(self.next(se)), self.vert(se)) > 0
    }

    /// Blender `dc_tri_valid`: whether `se` is above the base.
    fn dc_tri_valid(&self, se: Id, basel: Id, basel_sym: Id) -> bool {
        self.orient_v(
            self.vert(self.next(se)),
            self.vert(basel_sym),
            self.vert(basel),
        ) > 0
    }

    /// Blender `dc_tri`: Delaunay triangulation of the lexicographically sorted `sites`. Returns
    /// the counter-clockwise hull symedges at the leftmost and rightmost sites.
    fn dc_tri(&mut self, sites: &[Id]) -> (Id, Id) {
        let n = sites.len();
        if n <= 1 {
            return (NIL, NIL);
        }
        if n <= 3 {
            return self.dc_tri_base(sites);
        }
        let n2 = n / 2;
        let (ldo, ldi) = self.dc_tri(&sites[..n2]);
        let (rdi, rdo) = self.dc_tri(&sites[n2..]);
        self.dc_merge(ldo, ldi, rdi, rdo)
    }

    /// Blender `dc_tri` base case, for two or three sites.
    fn dc_tri_base(&mut self, sites: &[Id]) -> (Id, Id) {
        let v1 = sites[0];
        let v2 = sites[1];
        let outer = self.outer;
        let ea = s0(self.add_edge(v1, v2, outer, outer));
        let ea_sym = ea + 1;
        self.sym_edge_mut(ea).next = ea_sym;
        self.sym_edge_mut(ea_sym).next = ea;
        self.sym_edge_mut(ea).rot = ea;
        self.sym_edge_mut(ea_sym).rot = ea_sym;
        if sites.len() == 2 {
            return (ea, ea_sym);
        }
        let v3 = sites[2];
        let eb = s0(self.add_vert_to_symedge_edge(v3, ea_sym));
        let orient = self.orient_v(v1, v2, v3);
        match orient {
            1.. => {
                self.add_diagonal(eb, ea);
                (ea, eb)
            }
            ..=-1 => {
                self.add_diagonal(ea, eb);
                (self.rot(ea), self.rot(eb))
            }
            0 => (ea, eb),
        }
    }

    /// Blender `dc_tri` merge step: joins the left and right triangulations along their lower
    /// common tangent, then bubbles the base up while deleting edges that fail the circle test.
    fn dc_merge(&mut self, mut ldo: Id, mut ldi: Id, mut rdi: Id, mut rdo: Id) -> (Id, Id) {
        loop {
            if self.vert_left_of_symedge(self.vert(rdi), ldi) {
                ldi = self.next(ldi);
            } else if self.vert_right_of_symedge(self.vert(ldi), rdi) {
                rdi = self.rot(self.sym(rdi));
            } else {
                break;
            }
        }

        let rdi_prev_next = self.next(self.sym(rdi));
        let ebasel = self.connect_separate_parts(rdi_prev_next, ldi);
        let mut basel = s0(ebasel);
        let mut basel_sym = basel + 1;
        if self.vert(ldi) == self.vert(ldo) {
            ldo = basel_sym;
        }
        if self.vert(rdi) == self.vert(rdo) {
            rdo = basel;
        }

        loop {
            let mut lcand = self.rot(basel_sym);
            let mut rcand = self.next(basel_sym);
            if self.dc_tri_valid(lcand, basel, basel_sym) {
                while self.incircle_v(
                    self.vert(basel_sym),
                    self.vert(basel),
                    self.vert(self.next(lcand)),
                    self.vert(self.next(self.rot(lcand))),
                ) > 0
                {
                    let t = self.rot(lcand);
                    let victim = self.sym(lcand);
                    self.delete_edge(victim);
                    lcand = t;
                }
            }
            if self.dc_tri_valid(rcand, basel, basel_sym) {
                while self.incircle_v(
                    self.vert(basel_sym),
                    self.vert(basel),
                    self.vert(self.next(rcand)),
                    self.vert(self.next(self.next(self.sym(rcand)))),
                ) > 0
                {
                    let t = self.next(self.sym(rcand));
                    self.delete_edge(rcand);
                    rcand = t;
                }
            }

            let valid_lcand = self.dc_tri_valid(lcand, basel, basel_sym);
            let valid_rcand = self.dc_tri_valid(rcand, basel, basel_sym);
            if !valid_lcand && !valid_rcand {
                break;
            }
            let take_rcand = !valid_lcand
                || (valid_rcand
                    && self.incircle_v(
                        self.vert(self.next(lcand)),
                        self.vert(lcand),
                        self.vert(rcand),
                        self.vert(self.next(rcand)),
                    ) > 0);
            let ebasel = if take_rcand {
                let from = self.next(rcand);
                self.add_diagonal(from, basel_sym)
            } else {
                let from = self.next(basel_sym);
                let to = self.sym(lcand);
                self.add_diagonal(from, to)
            };
            basel = s0(ebasel);
            basel_sym = basel + 1;
        }
        (ldo, rdo)
    }

    /// Blender `initial_triangulation`: Delaunay triangulation of all input vertices, with
    /// coincident vertices merged onto the lowest input index.
    fn initial_triangulation(&mut self) {
        let n = self.verts.len();
        if n <= 1 {
            return;
        }
        let mut sites: Vec<Id> = (0..n).map(to_id).collect();
        sites.sort_by(|&a, &b| self.compare_sites(a, b));
        self.find_site_merges(&sites);
        let kept: Vec<Id> = sites
            .into_iter()
            .filter(|&s| self.verts[ix(s)].merge_to == NIL)
            .collect();
        if kept.is_empty() {
            return;
        }
        self.dc_tri(&kept);
    }

    /// Lexicographic order on exact coordinates, then input index (Blender `site_lexicographic_sort`).
    fn compare_sites(&self, a: Id, b: Id) -> Ordering {
        let ca = &self.co(a).exact;
        let cb = &self.co(b).exact;
        ca[0]
            .cmp(&cb[0])
            .then_with(|| ca[1].cmp(&cb[1]))
            .then_with(|| a.cmp(&b))
    }

    /// Blender `find_site_merges`: in sorted order, every site equal to the group head merges into it.
    fn find_site_merges(&mut self, sites: &[Id]) {
        let n = sites.len();
        let mut i = 0;
        while i + 1 < n {
            let head = sites[i];
            let mut j = i + 1;
            while j < n && self.co(sites[j]).exact == self.co(head).exact {
                self.verts[ix(sites[j])].merge_to = head;
                j += 1;
            }
            if j - i > 1 {
                i = j - 1;
            }
            i += 1;
        }
    }

    /// Blender `re_delaunay_triangulate`: re-triangulates the pseudo-polygon on the left of `se`
    /// so that it satisfies the Delaunay condition (Anglada's method).
    fn re_delaunay(&mut self, se: Id) {
        let outer = self.outer;
        if self.face(se) == outer || self.face(self.sym(se)) == outer {
            return;
        }
        let mut count = 1;
        let mut ss = self.next(se);
        while ss != se {
            count += 1;
            ss = self.next(ss);
        }
        if count <= 3 {
            return;
        }
        let first = self.next(self.next(se));
        let a = self.vert(se);
        let b = self.vert(self.next(se));
        let mut c = self.vert(first);
        let mut cse = first;
        let mut ss = self.next(first);
        while ss != se {
            let v = self.vert(ss);
            if self.incircle_v(a, b, c, v) > 0 {
                c = v;
                cse = ss;
            }
            ss = self.next(ss);
        }

        let mut ebc = None;
        let mut eca = None;
        if !self.exists_edge(b, c) {
            let from = self.next(se);
            ebc = Some(self.add_diagonal(from, cse));
        }
        if !self.exists_edge(c, a) {
            eca = Some(self.add_diagonal(cse, se));
        }
        if let Some(e) = ebc {
            self.re_delaunay(s0(e) + 1);
        }
        if let Some(e) = eca {
            self.re_delaunay(s0(e) + 1);
        }
    }

    /// Blender `fill_crossdata_for_through_vert`: the next crossing is vertex `v`.
    fn fill_crossdata_for_through_vert(
        &self,
        v: Id,
        cd_out: Id,
        cd: &mut CrossData,
        cd_next: &mut CrossData,
    ) {
        cd_next.lambda = Rat::zero();
        cd_next.vert = v;
        cd_next.in_ = NIL;
        cd_next.out = NIL;
        if cd.lambda.is_zero() {
            cd.out = cd_out;
        } else {
            let mut se = self.sym(cd.in_);
            if self.vert(se) != v {
                se = self.next(se);
                if self.vert(se) != v {
                    se = self.next(se);
                }
            }
            cd_next.in_ = se;
        }
    }

    /// Blender `fill_crossdata_for_intersect`: the ray from `curco` to `v2` crosses symedge `t`.
    fn fill_crossdata_for_intersect(
        &self,
        curco: &Rat2,
        v2: Id,
        t: Id,
        cd: &mut CrossData,
        cd_next: &mut CrossData,
    ) {
        let va = self.vert(t);
        let vb = self.vert(self.next(t));
        let se_vcvb = self.sym(self.next(t));
        let se_vcva = self.next(self.next(t));
        let (kind, lambda) = isect_seg_seg(
            &self.co(va).exact,
            &self.co(vb).exact,
            curco,
            &self.co(v2).exact,
        );
        let one = Rat::one();
        match kind {
            IsectKind::Exact if lambda.is_zero() => {
                self.fill_crossdata_for_through_vert(va, se_vcva, cd, cd_next);
            }
            IsectKind::Exact if lambda == one => {
                self.fill_crossdata_for_through_vert(vb, se_vcvb, cd, cd_next);
            }
            IsectKind::Cross | IsectKind::Exact => {
                *cd_next = CrossData {
                    lambda,
                    vert: NIL,
                    in_: t,
                    out: NIL,
                };
                if cd.lambda.is_zero() {
                    cd.out = se_vcva;
                }
            }
            IsectKind::Miss => {
                // Blender asserts that exact arithmetic never misses here, then uses the nearer end.
                let half = &one / (&one + &one);
                if lambda <= half {
                    self.fill_crossdata_for_through_vert(va, se_vcva, cd, cd_next);
                } else {
                    self.fill_crossdata_for_through_vert(vb, se_vcvb, cd, cd_next);
                }
            }
            IsectKind::Colinear => {
                if approx_dist_sq(&self.co(va).approx, &self.co(v2).approx)
                    <= approx_dist_sq(&self.co(vb).approx, &self.co(v2).approx)
                {
                    self.fill_crossdata_for_through_vert(va, se_vcva, cd, cd_next);
                } else {
                    self.fill_crossdata_for_through_vert(vb, se_vcvb, cd, cd_next);
                }
            }
        }
    }

    /// Blender `get_next_crossing_from_vert`: rotates around vertex `cd.vert` to find where the
    /// ray to `v2` leaves it. Returns `false` if no such position exists.
    fn get_next_crossing_from_vert(
        &self,
        cd: &mut CrossData,
        cd_next: &mut CrossData,
        v2: Id,
    ) -> bool {
        let vcur = cd.vert;
        let tstart = self.verts[ix(vcur)].symedge;
        let mut t = tstart;
        loop {
            let va = self.vert(self.next(t));
            let vb = self.vert(self.next(self.next(t)));
            let orient1 = self.orient_v(self.vert(t), va, v2);
            if orient1 == 0 && in_line(self.co(vcur), self.co(va), self.co(v2)) {
                self.fill_crossdata_for_through_vert(va, t, cd, cd_next);
                return true;
            }
            if self.face(t) != self.outer {
                let orient2 = self.orient_v(vcur, vb, v2);
                if orient1 > 0 && orient2 < 0 {
                    t = self.next(t);
                    self.fill_crossdata_for_intersect(&self.co(vcur).exact, v2, t, cd, cd_next);
                    return true;
                }
            }
            t = self.rot(t);
            if t == tstart {
                return false;
            }
        }
    }

    /// Blender `get_next_crossing_from_edge`: the path passes through the interior of the edge
    /// crossed by `cd`, so look at the triangle on the far side.
    fn get_next_crossing_from_edge(&self, cd: &mut CrossData, cd_next: &mut CrossData, v2: Id) {
        let va = self.vert(cd.in_);
        let vb = self.vert(self.next(cd.in_));
        let curco = FatCo::new(interpolate(
            &self.co(va).exact,
            &self.co(vb).exact,
            &cd.lambda,
        ));
        let se_ac = self.next(self.sym(cd.in_));
        let vc = self.vert(self.next(se_ac));
        let orient = filtered_orient2d(&curco, self.co(v2), self.co(vc));
        match orient {
            ..=-1 => {
                let t = self.next(se_ac);
                self.fill_crossdata_for_intersect(&curco.exact, v2, t, cd, cd_next);
            }
            1.. => {
                self.fill_crossdata_for_intersect(&curco.exact, v2, se_ac, cd, cd_next);
            }
            0 => {
                *cd_next = CrossData {
                    lambda: Rat::zero(),
                    vert: vc,
                    in_: self.next(se_ac),
                    out: NIL,
                };
            }
        }
    }

    /// Blender `add_edge_constraint`: inserts the constraint `v1 -> v2`, splitting crossed
    /// constrained edges and re-triangulating around the new edges. Returns the edges that
    /// now make up the constraint, in order.
    fn add_edge_constraint(&mut self, v1: Id, v2: Id, input_id: u32) -> Vec<Id> {
        if v1 == v2 {
            return Vec::new();
        }
        let existing = self.find_symedge_between_verts(v1, v2);
        if existing != NIL {
            let e = edge_of(existing);
            self.edge_ids[ix(e)].insert(input_id);
            return vec![e];
        }

        self.visit_count += 1;
        let visit = self.visit_count;
        let mut crossings = vec![CrossData {
            lambda: Rat::zero(),
            vert: v1,
            in_: NIL,
            out: NIL,
        }];
        loop {
            let n = crossings.len();
            if crossings[n - 1].vert == v2 {
                break;
            }
            crossings.push(CrossData::empty());
            let (head, tail) = crossings.split_at_mut(n);
            let cd = &mut head[n - 1];
            let cd_next = &mut tail[0];
            let ok = if cd.lambda.is_zero() {
                self.get_next_crossing_from_vert(cd, cd_next, v2)
            } else {
                self.get_next_crossing_from_edge(cd, cd_next, v2);
                true
            };
            if !ok || crossings.len() == UNREASONABLY_LARGE_CROSSINGS {
                return Vec::new();
            }
            if crossings[n].lambda.is_zero() {
                let v = crossings[n].vert;
                if self.verts[ix(v)].visit_index == visit {
                    return Vec::new();
                }
                self.verts[ix(v)].visit_index = visit;
            }
        }

        let ncross = crossings.len();
        let deleted = Rat::zero() - Rat::one();

        // Discard crossings that the path passes through as vertices, preferring the direct route.
        let mut i = 2;
        while i < ncross {
            if !crossings[i].lambda.is_zero() {
                i += 1;
                continue;
            }
            let v = crossings[i].vert;
            let mut j = i - 1;
            while j > 0 {
                let prev = &crossings[j];
                let stop = if prev.lambda.is_zero() {
                    prev.vert != v
                } else {
                    self.vert(prev.in_) != v && self.vert(self.next(prev.in_)) != v
                };
                if stop {
                    break;
                }
                crossings[j].lambda = deleted.clone();
                j -= 1;
            }
            if j < i - 1 {
                if crossings[j].lambda.is_zero() {
                    let se = self.find_symedge_between_verts(crossings[j].vert, v);
                    if se == NIL {
                        return Vec::new();
                    }
                    crossings[j].out = se;
                    crossings[i].in_ = NIL;
                } else {
                    let face = self.face(self.sym(crossings[j].in_));
                    let se = self.find_symedge_with_face(v, face);
                    if se == NIL {
                        return Vec::new();
                    }
                    crossings[i].in_ = se;
                }
            }
            i += 1;
        }

        // Split constrained edges at the crossing points.
        for cd in &mut crossings {
            if !cd.lambda.is_zero() && !cd.lambda.is_negative() && self.is_constrained(cd.in_) {
                let split = self.split_edge(cd.in_, &cd.lambda);
                cd.vert = self.vert(s0(split));
            }
        }

        // Delete crossed unconstrained edges.
        for cd in &crossings {
            if !cd.lambda.is_zero() && !cd.lambda.is_negative() && !self.is_constrained(cd.in_) {
                self.delete_edge(cd.in_);
            }
        }

        // Thread the constraint through the surviving crossings.
        let mut edge_list = Vec::new();
        let mut tstart = crossings[0].out;
        for (i, cd) in crossings.iter().enumerate().skip(1) {
            if cd.lambda.is_negative() {
                continue;
            }
            let mut t = NIL;
            let mut tnext = NIL;
            if cd.lambda.is_zero() {
                t = cd.in_;
                tnext = cd.out;
                if t == NIL {
                    // The previous live crossing was a vertex, so the constraint already exists.
                    let mut j = i - 1;
                    while j > 0 && crossings[j].lambda.is_negative() {
                        j -= 1;
                    }
                    let out = crossings[j].out;
                    if out != NIL {
                        let e = edge_of(out);
                        self.edge_ids[ix(e)].insert(input_id);
                        edge_list.push(e);
                    }
                }
            } else if self.is_constrained(cd.in_) {
                t = self.verts[ix(cd.vert)].symedge;
                tnext = self.next(self.sym(t));
            }
            if t != NIL {
                let e = if self.vert(self.next(tstart)) == self.vert(t) {
                    edge_of(tstart)
                } else {
                    self.add_diagonal(tstart, t)
                };
                self.edge_ids[ix(e)].insert(input_id);
                edge_list.push(e);
                self.re_delaunay(s0(e));
                self.re_delaunay(s0(e) + 1);
            }
            if i < ncross - 1 && tnext != NIL {
                tstart = tnext;
            }
        }
        edge_list
    }

    /// Blender `add_edge_constraints`: inserts every loose input edge. Sets `face_edge_offset`
    /// to the number of loose edges.
    fn add_edge_constraints(&mut self, edges: &[(u32, u32)]) {
        let nv = self.input_vert_num;
        for (i, &(a, b)) in edges.iter().enumerate() {
            if a as usize >= nv || b as usize >= nv {
                continue;
            }
            let v1 = self.resolve_merge(a);
            let v2 = self.resolve_merge(b);
            self.add_edge_constraint(v1, v2, to_u32(i));
        }
        self.face_edge_offset = to_u32(edges.len());
    }

    /// Blender `get_vert_resolve_merge`: the vertex an input index ends up as.
    fn resolve_merge(&self, input: u32) -> Id {
        let merge = self.verts[ix(input)].merge_to;
        if merge == NIL { input } else { merge }
    }

    /// Blender `add_face_ids`: labels every face inside the input face with `face_id`, flooding
    /// without crossing the face's own boundary (edge ids in `fedge_start..=fedge_end`).
    fn add_face_ids(&mut self, start: Id, face_id: u32, fedge_start: u32, fedge_end: u32) {
        self.visit_count += 1;
        let visit = self.visit_count;
        let mut stack = vec![start];
        while let Some(se) = stack.pop() {
            let face = self.face(se);
            if self.faces[ix(face)].visit_index == visit {
                continue;
            }
            self.faces[ix(face)].visit_index = visit;
            self.faces[ix(face)].input_ids.insert(face_id);
            let se_start = se;
            let mut s = self.next(se);
            while s != se_start {
                let on_boundary = self.edge_ids[ix(edge_of(s))]
                    .range(fedge_start..=fedge_end)
                    .next()
                    .is_some();
                if !on_boundary {
                    let se_sym = self.sym(s);
                    let face_other = self.face(se_sym);
                    if self.faces[ix(face_other)].visit_index != visit {
                        stack.push(se_sym);
                    }
                }
                s = self.next(s);
            }
        }
    }

    /// Blender `add_face_constraints` for `CDT_INSIDE`, which needs no winding or
    /// polygon-boundary bookkeeping.
    fn add_face_constraints(&mut self, faces: &[Vec<u32>]) {
        let nv = self.input_vert_num;
        let maxflen = to_u32(faces.iter().map(Vec::len).max().unwrap_or(0));
        self.face_edge_offset = maxflen.max(self.face_edge_offset);
        let offset = self.face_edge_offset;
        // Blender keeps this across faces, so a face whose edges all degenerate reuses the last one.
        let mut face_symedge0 = NIL;
        for (f, face) in faces.iter().enumerate() {
            if face.len() <= 2 {
                continue;
            }
            let fedge_start = (to_u32(f) + 1).wrapping_mul(offset);
            for (i, &iv1) in face.iter().enumerate() {
                let face_edge_id = fedge_start.wrapping_add(to_u32(i));
                let iv2 = face[(i + 1) % face.len()];
                if iv1 as usize >= nv || iv2 as usize >= nv {
                    continue;
                }
                let v1 = self.resolve_merge(iv1);
                let v2 = self.resolve_merge(iv2);
                let edges = self.add_edge_constraint(v1, v2, face_edge_id);
                if let Some(&e) = edges.first() {
                    let s = s0(e);
                    face_symedge0 = if self.vert(s) == v1 { s } else { s + 1 };
                }
            }
            let fedge_end = fedge_start.wrapping_add(to_u32(face.len()) - 1);
            if face_symedge0 != NIL {
                self.add_face_ids(face_symedge0, to_u32(f), fedge_start, fedge_end);
            }
        }
    }

    /// Blender `dissolve_symedge`: deletes an edge while keeping face and outer-face symedges valid.
    fn dissolve_symedge(&mut self, se: Id) {
        let outer = self.outer;
        let mut se = se;
        let mut symse = self.sym(se);
        if self.face(symse) == outer {
            se = self.sym(se);
            symse = self.sym(se);
        }
        let outer_se = self.faces[ix(outer)].symedge;
        if outer_se == se || outer_se == symse {
            let after = self.next(self.next(se));
            self.faces[ix(outer)].symedge = if after == se { NIL } else { after };
        } else {
            let v1_isolated = self.next(symse) == se;
            let v2_isolated = self.next(se) == symse;
            let fse = self.face(se);
            if self.faces[ix(fse)].symedge == se {
                self.faces[ix(fse)].symedge = if v2_isolated {
                    self.prev(se)
                } else {
                    self.next(se)
                };
            }
            let fsym = self.face(symse);
            if self.faces[ix(fsym)].symedge == symse {
                self.faces[ix(fsym)].symedge = if v1_isolated {
                    self.prev(symse)
                } else {
                    self.next(symse)
                };
            }
        }
        self.delete_edge(se);
    }

    /// Blender `remove_outer_edges_until_constraints`: flood from the outer face across
    /// unconstrained edges, dissolving the edges between flooded faces. Everything not reached
    /// this way is inside the constraints and is kept.
    fn remove_outer_edges_until_constraints(&mut self) {
        self.visit_count += 1;
        let visit = self.visit_count;
        let outer = self.outer;
        self.faces[ix(outer)].visit_index = visit;

        let mut fstack: Vec<Id> = Vec::new();
        let se_start = self.faces[ix(outer)].symedge;
        if se_start != NIL {
            let mut se = se_start;
            loop {
                if !self.is_constrained(se) {
                    let fsym = self.face(self.sym(se));
                    if self.faces[ix(fsym)].visit_index != visit {
                        fstack.push(fsym);
                    }
                }
                se = self.next(se);
                if se == se_start {
                    break;
                }
            }
        }

        while let Some(f) = fstack.pop() {
            if self.faces[ix(f)].visit_index == visit {
                continue;
            }
            self.faces[ix(f)].visit_index = visit;
            let se_start = self.faces[ix(f)].symedge;
            if se_start == NIL {
                continue;
            }
            let mut to_dissolve: Vec<Id> = Vec::new();
            let mut se = se_start;
            loop {
                if !self.is_constrained(se) {
                    let fsym = self.face(self.sym(se));
                    if self.faces[ix(fsym)].visit_index == visit {
                        to_dissolve.push(se);
                    } else {
                        fstack.push(fsym);
                    }
                }
                se = self.next(se);
                if se == se_start {
                    break;
                }
            }
            while let Some(s) = to_dissolve.pop() {
                if self.next(s) != NIL {
                    self.dissolve_symedge(s);
                }
            }
        }
    }

    /// Blender `prepare_cdt_for_output` for `CDT_INSIDE`.
    fn prepare_for_output(&mut self) {
        if self.edge_ids.is_empty() {
            return;
        }
        for e in 0..self.edge_ids.len() {
            let e = to_id(e);
            if self.is_deleted_edge(e) {
                continue;
            }
            for se in [s0(e), s0(e) + 1] {
                let face = self.face(se);
                if self.faces[ix(face)].symedge == NIL {
                    self.faces[ix(face)].symedge = se;
                }
            }
        }
        self.remove_outer_edges_until_constraints();
    }

    /// Blender `get_cdt_output`.
    fn output(&mut self) -> CdtOutput {
        let mut out = CdtOutput {
            face_edge_offset: self.face_edge_offset,
            ..CdtOutput::default()
        };
        let nverts = self.verts.len();
        let mut vert_map: Vec<Id> = vec![NIL; nverts];
        let mut nout: Id = 0;
        for (i, v) in self.verts.iter().enumerate() {
            if v.merge_to == NIL {
                vert_map[i] = nout;
                nout += 1;
            }
        }
        if nout == 0 {
            return out;
        }

        let merged: Vec<(usize, Id)> = self
            .verts
            .iter()
            .enumerate()
            .filter(|(_, v)| v.merge_to != NIL)
            .map(|(i, v)| (i, v.merge_to))
            .collect();
        for (i, target) in merged {
            if i < self.input_vert_num {
                self.verts[ix(target)].input_ids.insert(to_u32(i));
            }
            vert_map[i] = vert_map[ix(target)];
        }

        for (i, v) in self.verts.iter().enumerate() {
            if v.merge_to != NIL {
                continue;
            }
            out.verts.push(v.co.exact.clone());
            let mut orig = Vec::new();
            if i < self.input_vert_num {
                orig.push(to_u32(i));
            }
            orig.extend(v.input_ids.iter().copied());
            out.vert_orig.push(orig);
        }

        for (e, ids) in self.edge_ids.iter().enumerate() {
            let se = s0(to_id(e));
            if self.sym_edge(se).next == NIL {
                continue;
            }
            let a = vert_map[ix(self.vert(se))];
            let b = vert_map[ix(self.vert(se + 1))];
            out.edges.push((a, b));
            out.edge_orig.push(ids.iter().copied().collect());
        }

        let outer = self.outer;
        for (f, face) in self.faces.iter().enumerate() {
            if face.deleted || to_id(f) == outer || face.symedge == NIL {
                continue;
            }
            let start = face.symedge;
            let mut poly = Vec::new();
            let mut se = start;
            loop {
                poly.push(vert_map[ix(self.vert(se))]);
                se = self.next(se);
                if se == start {
                    break;
                }
            }
            push_face(&poly, &face.input_ids, &mut out);
        }
        out
    }
}

/// Appends the triangles of one output polygon (Blender's consumers expect triangles).
fn push_face(poly: &[Id], orig: &BTreeSet<u32>, out: &mut CdtOutput) {
    if poly.len() < 3 {
        return;
    }
    debug_assert!(poly.len() == 3, "CDT_INSIDE faces are triangles");
    for pair in poly[1..].windows(2) {
        out.faces.push([poly[0], pair[0], pair[1]]);
        out.face_orig.push(orig.iter().copied().collect());
    }
}

/// Constrained Delaunay triangulation of `input`, keeping the part inside its faces
/// (Blender `delaunay_2d_calc<mpq_class>(input, CDT_INSIDE)`).
#[must_use]
pub(crate) fn delaunay_2d_calc_inside(input: &CdtInput) -> CdtOutput {
    let mut cdt = Cdt::new(input.verts.len());
    for co in &input.verts {
        cdt.add_vert(co.clone());
    }
    cdt.initial_triangulation();
    cdt.add_edge_constraints(&input.edges);
    cdt.add_face_constraints(&input.faces);
    cdt.prepare_for_output();
    cdt.output()
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::super::exact::rat_from_f64;
    use super::*;

    fn r(n: i64) -> Rat {
        Rat::from_integer(n.into())
    }

    fn p(x: i64, y: i64) -> Rat2 {
        [r(x), r(y)]
    }

    /// Exact doubled signed area of triangle `a, b, c`.
    fn doubled_area(a: &Rat2, b: &Rat2, c: &Rat2) -> Rat {
        (&b[0] - &a[0]) * (&c[1] - &a[1]) - (&b[1] - &a[1]) * (&c[0] - &a[0])
    }

    fn tri_doubled_area(out: &CdtOutput, t: &[u32; 3]) -> Rat {
        doubled_area(
            &out.verts[t[0] as usize],
            &out.verts[t[1] as usize],
            &out.verts[t[2] as usize],
        )
    }

    fn total_doubled_area(out: &CdtOutput) -> Rat {
        out.faces
            .iter()
            .fold(Rat::zero(), |acc, t| acc + tri_doubled_area(out, t))
    }

    fn edge_present(out: &CdtOutput, a: u32, b: u32) -> bool {
        out.edges
            .iter()
            .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
    }

    fn square() -> Vec<Rat2> {
        vec![p(0, 0), p(1, 0), p(1, 1), p(0, 1)]
    }

    /// Every non-constrained edge shared by two output triangles must satisfy the Delaunay
    /// condition: the opposite vertex of one triangle is not strictly inside the other's circumcircle.
    fn assert_locally_delaunay(out: &CdtOutput) {
        let key = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };
        let constrained: HashSet<(u32, u32)> = out
            .edges
            .iter()
            .zip(&out.edge_orig)
            .filter(|(_, orig)| !orig.is_empty())
            .map(|(&(a, b), _)| key(a, b))
            .collect();
        let mut uses: HashMap<(u32, u32), Vec<(usize, u32)>> = HashMap::new();
        for (t, tri) in out.faces.iter().enumerate() {
            for (a, b, opp) in [
                (tri[0], tri[1], tri[2]),
                (tri[1], tri[2], tri[0]),
                (tri[2], tri[0], tri[1]),
            ] {
                uses.entry(key(a, b)).or_default().push((t, opp));
            }
        }
        for (edge, list) in &uses {
            if constrained.contains(edge) || list.len() != 2 {
                continue;
            }
            let (t, _) = list[0];
            let (_, opp) = list[1];
            let tri = out.faces[t];
            let inside = exact_incircle(
                &out.verts[tri[0] as usize],
                &out.verts[tri[1] as usize],
                &out.verts[tri[2] as usize],
                &out.verts[opp as usize],
            );
            assert!(
                inside <= 0,
                "non-constrained edge {edge:?} violates the Delaunay condition"
            );
        }
    }

    #[test]
    fn unit_square_face_gives_two_ccw_triangles() {
        let input = CdtInput {
            verts: square(),
            edges: vec![],
            faces: vec![vec![0, 1, 2, 3]],
        };
        let out = delaunay_2d_calc_inside(&input);
        assert_eq!(out.faces.len(), 2);
        assert_eq!(out.verts.len(), 4);
        for t in &out.faces {
            assert!(tri_doubled_area(&out, t).is_positive());
        }
        assert_eq!(total_doubled_area(&out), r(2));
        assert_locally_delaunay(&out);
    }

    #[test]
    fn constraint_diagonal_is_kept() {
        let input = CdtInput {
            verts: square(),
            edges: vec![(0, 2)],
            faces: vec![vec![0, 1, 2, 3]],
        };
        let out = delaunay_2d_calc_inside(&input);
        assert_eq!(out.faces.len(), 2);
        assert!(edge_present(&out, 0, 2));
        assert_eq!(total_doubled_area(&out), r(2));
    }

    #[test]
    fn interior_segment_splits_triangle() {
        // A(0,0) B(4,0) C(0,4) with P(0,1) on AC and Q(3,1) on BC; the chord PQ splits the triangle
        // into CPQ (doubled area 9) and the quadrilateral ABQP (doubled area 7).
        let input = CdtInput {
            verts: vec![p(0, 0), p(4, 0), p(0, 4), p(0, 1), p(3, 1)],
            edges: vec![(3, 4)],
            faces: vec![vec![0, 1, 2]],
        };
        let out = delaunay_2d_calc_inside(&input);
        assert_eq!(out.verts.len(), 5);
        assert_eq!(out.faces.len(), 3);
        assert!(edge_present(&out, 3, 4));
        assert_eq!(total_doubled_area(&out), r(16));
        assert_locally_delaunay(&out);
    }

    #[test]
    fn crossing_constraints_split_at_exact_intersection() {
        let input = CdtInput {
            verts: square(),
            edges: vec![(0, 2), (1, 3)],
            faces: vec![vec![0, 1, 2, 3]],
        };
        let out = delaunay_2d_calc_inside(&input);
        // The diagonals meet at the created point (1/2, 1/2), which is output vertex 4.
        assert_eq!(out.verts.len(), 5);
        let half = Rat::one() / (Rat::one() + Rat::one());
        assert_eq!(out.verts[4], [half.clone(), half]);
        assert_eq!(out.vert_orig[4], [] as [u32; 0]);
        assert_eq!(out.faces.len(), 4);
        assert_eq!(total_doubled_area(&out), r(2));
    }

    #[test]
    fn shared_edge_faces_label_their_triangles() {
        let input = CdtInput {
            verts: square(),
            edges: vec![],
            faces: vec![vec![0, 1, 2], vec![0, 2, 3]],
        };
        let out = delaunay_2d_calc_inside(&input);
        assert_eq!(out.faces.len(), 2);
        let mut origs = out.face_orig.clone();
        origs.sort();
        assert_eq!(origs, vec![vec![0], vec![1]]);
    }

    #[test]
    fn offset_below_f64_resolution_is_exact() {
        // C = (1 + 2^-80, 1) rounds to (1, 1) in f64, which would merge it with E = (1, 1).
        // Exactly, E lies on the edge C -> D, so the pentagon A B C E D has three triangles.
        let eps = rat_from_f64(2f64.powi(-80));
        let input = CdtInput {
            verts: vec![p(0, 0), p(1, 0), [r(1) + &eps, r(1)], p(0, 1), p(1, 1)],
            edges: vec![],
            faces: vec![vec![0, 1, 2, 3]],
        };
        let out = delaunay_2d_calc_inside(&input);
        assert_eq!(out.verts.len(), 5);
        assert_eq!(out.faces.len(), 3);
        for t in &out.faces {
            assert!(tri_doubled_area(&out, t).is_positive());
        }
        assert_eq!(total_doubled_area(&out), r(2) + &eps);
    }

    #[test]
    fn coincident_input_vertices_merge() {
        let input = CdtInput {
            verts: vec![p(0, 0), p(1, 0), p(0, 1), p(1, 0)],
            edges: vec![],
            faces: vec![vec![0, 1, 2]],
        };
        let out = delaunay_2d_calc_inside(&input);
        assert_eq!(out.verts.len(), 3);
        assert_eq!(out.faces.len(), 1);
        assert_eq!(out.vert_orig[1], vec![1, 3]);
    }

    #[test]
    fn grid_with_interior_points_is_locally_delaunay() {
        let input = CdtInput {
            verts: vec![
                p(0, 0),
                p(4, 0),
                p(4, 4),
                p(0, 4),
                p(1, 1),
                p(3, 1),
                p(2, 2),
                p(1, 3),
                p(3, 3),
            ],
            edges: vec![],
            faces: vec![vec![0, 1, 2, 3]],
        };
        let out = delaunay_2d_calc_inside(&input);
        // 2n - h - 2 triangles for n = 9 points with h = 4 hull vertices.
        assert_eq!(out.faces.len(), 12);
        for t in &out.faces {
            assert!(tri_doubled_area(&out, t).is_positive());
        }
        assert_eq!(total_doubled_area(&out), r(32));
        assert_locally_delaunay(&out);
    }

    #[test]
    fn incircle_sign_convention() {
        // Unit circle through CCW (1,0), (0,1), (-1,0).
        let a = p(1, 0);
        let b = p(0, 1);
        let c = p(-1, 0);
        assert_eq!(exact_incircle(&a, &b, &c, &p(0, 0)), 1);
        assert_eq!(exact_incircle(&a, &b, &c, &p(2, 0)), -1);
        assert_eq!(exact_incircle(&a, &b, &c, &p(0, -1)), 0);
    }
}
