//! Output polygons of the exact boolean (Blender `polymesh_from_trimesh_with_dissolve`).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use glam::DVec3;

use super::imesh::{Arena, NO_INDEX, Tri};

/// An output polygon: a region of one input face, bounded by input edges and cut segments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Poly {
    /// Arena vertex indices in ring order.
    pub verts: Vec<u32>,
    /// Input face index (global over joined operands) the polygon came from.
    pub orig: i32,
    /// Per ring edge `i` (from `verts[i]` to `verts[i + 1]`): input edge index, or [`NO_INDEX`].
    pub edge_orig: Vec<i32>,
}

impl Poly {
    /// Whether ring edge `i` is a new cut edge rather than an input edge.
    #[cfg(test)]
    #[must_use]
    pub fn is_new_edge(&self, i: usize) -> bool {
        self.edge_orig.get(i).is_none_or(|&e| e == NO_INDEX)
    }
}

/// Blender `polymesh_from_trimesh_with_dissolve`: merges the output triangles of each input face
/// into polygons by dissolving triangulation edges, then removes non-input vertices that became
/// valence-2 corners on a straight line.
///
/// `tris` is the intersected triangle mesh in Blender's `tm_out` order. The result keeps Blender's
/// order: input faces ascending, then the merge order inside each input face.
#[must_use]
pub(crate) fn polymesh_from_tris(arena: &Arena, tris: &[Tri]) -> Vec<Poly> {
    // Blender `face_output_tris`: group by input face, keeping triangle order within a group.
    let mut by_orig: BTreeMap<i32, Vec<Tri>> = BTreeMap::new();
    for tri in tris {
        by_orig.entry(tri.orig).or_default().push(*tri);
    }
    let mut polys = Vec::new();
    for group in by_orig.values() {
        polys.extend(merge_tris_for_face(arena, group));
    }
    dissolve_vertices(arena, polys)
}

/// Blender `MergeEdge`: an undirected triangulation edge, with `v1 < v2` by arena id.
#[derive(Clone, Copy, Debug)]
struct MergeEdge {
    /// Squared length, used to order dissolution (longest first).
    len_squared: f64,
    v1: u32,
    v2: u32,
    /// Merge-face index on the left of `v1 -> v2`, if any.
    left_face: Option<usize>,
    /// Merge-face index on the right of `v1 -> v2`, if any.
    right_face: Option<usize>,
    /// Input edge this edge lies on, or [`NO_INDEX`] for triangulation edges.
    orig: i32,
    /// Whether this edge may be removed by dissolution.
    dissolvable: bool,
}

/// Blender `MergeFace`: a polygon under construction during dissolution.
#[derive(Clone, Debug)]
struct MergeFace {
    vert: Vec<u32>,
    /// Per ring position, the index in [`FaceMergeState::edge`] of the edge leaving that vertex.
    edge: Vec<usize>,
    /// Face this one was merged into; `None` while it is still an output face.
    merge_to: Option<usize>,
    orig: i32,
}

/// Blender `FaceMergeState`.
#[derive(Default)]
struct FaceMergeState {
    face: Vec<MergeFace>,
    edge: Vec<MergeEdge>,
    /// Canonical `(min id, max id)` vertex pair to index in `edge`.
    edge_map: HashMap<(u32, u32), usize>,
}

/// Normal of a triangle from double coordinates (Blender `Face::populate_plane(false)`).
fn tri_normal(arena: &Arena, tri: &Tri) -> DVec3 {
    let v0 = arena.vert(tri.v[0]).co_f;
    let v1 = arena.vert(tri.v[1]).co_f;
    let v2 = arena.vert(tri.v[2]).co_f;
    (v0 - v2).cross(v1 - v2)
}

fn poly_from_tri(tri: &Tri) -> Poly {
    Poly {
        verts: tri.v.to_vec(),
        orig: tri.orig,
        edge_orig: tri.edge_orig.to_vec(),
    }
}

/// Blender `find_tris_common_edge`: `(i, j)` such that `t1` edge `i` is `t2` edge `j` reversed.
fn find_tris_common_edge(t1: &Tri, t2: &Tri) -> Option<(usize, usize)> {
    for i in 0..3 {
        for j in 0..3 {
            if t1.v[(i + 1) % 3] == t2.v[j] && t1.v[i] == t2.v[(j + 1) % 3] {
                return Some((i, j));
            }
        }
    }
    None
}

/// Blender's "quad with one diagonal remained unchanged" case of `merge_tris_for_face`.
///
/// shortcut: Blender checks the candidate ring against the input face (`cyclic_equal`), which
/// this API does not carry. All four boundary edges being input edges identifies the same quad,
/// but the ring starts at the first triangle's corner rather than at the input face's start.
/// Upgrade when output corner/UV attributes must match the input loop start exactly.
fn recover_quad(tri1: &Tri, tri2: &Tri) -> Option<Poly> {
    let (i0, j) = find_tris_common_edge(tri1, tri2)?;
    if tri1.edge_orig[i0] != NO_INDEX {
        return None;
    }
    let i1 = (i0 + 1) % 3;
    let i2 = (i0 + 2) % 3;
    let j1 = (j + 1) % 3;
    let j2 = (j + 2) % 3;
    // Ring: tri1[i1] -> tri1[i2] -> tri1[i0] -> tri2[j2]; its edges are tri1 edge i1, tri1 edge
    // i2, tri2 edge j1 and tri2 edge j2.
    let edge_orig = vec![
        tri1.edge_orig[i1],
        tri1.edge_orig[i2],
        tri2.edge_orig[j1],
        tri2.edge_orig[j2],
    ];
    if edge_orig.contains(&NO_INDEX) {
        return None;
    }
    Some(Poly {
        verts: vec![tri1.v[i1], tri1.v[i2], tri1.v[i0], tri2.v[j2]],
        orig: tri1.orig,
        edge_orig,
    })
}

/// Blender `init_face_merge_state`: the triangles of `tris` that face `norm`, with their edges.
fn init_face_merge_state(fms: &mut FaceMergeState, arena: &Arena, tris: &[Tri], norm: DVec3) {
    for tri in tris {
        if norm.dot(tri_normal(arena, tri)) <= 0.0 {
            continue;
        }
        let f = fms.face.len();
        fms.face.push(MergeFace {
            vert: tri.v.to_vec(),
            edge: Vec::with_capacity(3),
            merge_to: None,
            orig: tri.orig,
        });
        for i in 0..3 {
            let (a, b) = (tri.v[i], tri.v[(i + 1) % 3]);
            let key = if a < b { (a, b) } else { (b, a) };
            let me_index = if let Some(&idx) = fms.edge_map.get(&key) {
                idx
            } else {
                let idx = fms.edge.len();
                let orig = tri.edge_orig[i];
                let vec = arena.vert(key.1).co_f - arena.vert(key.0).co_f;
                fms.edge.push(MergeEdge {
                    len_squared: vec.length_squared(),
                    v1: key.0,
                    v2: key.1,
                    left_face: None,
                    right_face: None,
                    orig,
                    dissolvable: orig == NO_INDEX && !tri.is_intersect[i],
                });
                fms.edge_map.insert(key, idx);
                idx
            };
            let me = &mut fms.edge[me_index];
            if me.dissolvable && tri.edge_orig[i] != NO_INDEX {
                me.dissolvable = false;
                me.orig = tri.edge_orig[i];
            }
            if me.dissolvable && tri.is_intersect[i] {
                me.dissolvable = false;
            }
            // This face is left or right of the canonical edge, depending on its direction.
            let on_left = me.v1 == a;
            let occupied = if on_left {
                me.left_face.is_some()
            } else {
                me.right_face.is_some()
            };
            if occupied {
                // More than one triangle on the same side: tolerate it, never dissolve it.
                me.dissolvable = false;
            } else if on_left {
                me.left_face = Some(f);
            } else {
                me.right_face = Some(f);
            }
            fms.face[f].edge.push(me_index);
        }
    }
}

/// Blender `dissolve_leaves_valid_bmesh`: removing `me` must not leave a second shared edge
/// between the two faces, nor a repeated vertex.
fn dissolve_leaves_valid_bmesh(
    fms: &FaceMergeState,
    me_index: usize,
    left: usize,
    right: usize,
) -> bool {
    let me = fms.edge[me_index];
    let mf_left = &fms.face[left];
    let mf_right = &fms.face[right];
    let Some(a_start) = mf_left.edge.iter().position(|&e| e == me_index) else {
        return false;
    };
    let alen = mf_left.vert.len();
    // Is there another edge, not me, in A's face, whose right face is B's left?
    for k in 1..alen {
        let a_me = &fms.edge[mf_left.edge[(a_start + k) % alen]];
        if a_me.right_face == me.right_face {
            return false;
        }
    }
    // Is there a vert in A, not me.v1 or me.v2, that is also in B?
    mf_left
        .vert
        .iter()
        .all(|&a_v| a_v == me.v1 || a_v == me.v2 || !mf_right.vert.contains(&a_v))
}

/// Blender `splice_faces`: replaces `me` in face `left` by the rest of face `right`, and marks
/// `right` as merged into `left`.
fn splice_faces(fms: &mut FaceMergeState, me_index: usize, left: usize, right: usize) {
    let me = fms.edge[me_index];
    let (Some(a_start), Some(b_start)) = (
        fms.face[left].edge.iter().position(|&e| e == me_index),
        fms.face[right].edge.iter().position(|&e| e == me_index),
    ) else {
        return;
    };
    let a_vert = fms.face[left].vert.clone();
    let a_edge = fms.face[left].edge.clone();
    let b_vert = fms.face[right].vert.clone();
    let b_edge = fms.face[right].edge.clone();
    let (alen, blen) = (a_vert.len(), b_vert.len());
    let mut splice_vert = Vec::with_capacity(alen + blen);
    let mut splice_edge = Vec::with_capacity(alen + blen);
    for ai in 0..a_start {
        splice_vert.push(a_vert[ai]);
        splice_edge.push(a_edge[ai]);
    }
    // B's ring, from the edge after `me` around to the edge before it.
    for k in 1..blen {
        let bi = (b_start + k) % blen;
        let (bv, be) = (b_vert[bi], b_edge[bi]);
        splice_vert.push(bv);
        splice_edge.push(be);
        if bv == fms.edge[be].v1 {
            fms.edge[be].left_face = me.left_face;
        } else {
            fms.edge[be].right_face = me.left_face;
        }
    }
    for ai in a_start + 1..alen {
        splice_vert.push(a_vert[ai]);
        splice_edge.push(a_edge[ai]);
    }
    fms.face[right].merge_to = me.left_face;
    fms.face[left].vert = splice_vert;
    fms.face[left].edge = splice_edge;
    fms.edge[me_index].left_face = None;
    fms.edge[me_index].right_face = None;
}

/// libstdc++ `_S_threshold`: ranges up to this length are finished by insertion sort.
const SORT_THRESHOLD: usize = 16;

/// Sorts `a` as libstdc++'s `std::sort` does (GCC 14, `bits/stl_algo.h`, `bits/stl_heap.h`).
/// Blender's `do_dissolve` uses `std::ranges::sort`, which is unstable; on Linux (Blender 5.2.2)
/// that is this algorithm, so equal lengths keep Blender's order. `less(x, y)`: `x` sorts first.
/// Verified against the real libstdc++ `std::ranges::sort`, including the heap fallback.
fn libstdcxx_sort_by<T: Copy, F: Fn(T, T) -> bool>(a: &mut [T], less: &F) {
    if a.is_empty() {
        return;
    }
    let len = a.len();
    introsort_loop(a, 0, len, 2 * lg(len), less);
    final_insertion_sort(a, 0, len, less);
}

/// `__lg`: floor(log2(n)) for n >= 1.
fn lg(n: usize) -> usize {
    (usize::BITS - 1 - n.leading_zeros()) as usize
}

/// `__introsort_loop`: quicksort partitions until a range is at most `SORT_THRESHOLD` long, or
/// falls back to a heap sort once `depth` runs out.
fn introsort_loop<T: Copy, F: Fn(T, T) -> bool>(
    a: &mut [T],
    first: usize,
    mut last: usize,
    mut depth: usize,
    less: &F,
) {
    while last - first > SORT_THRESHOLD {
        if depth == 0 {
            partial_sort(a, first, last, last, less);
            return;
        }
        depth -= 1;
        let cut = unguarded_partition_pivot(a, first, last, less);
        introsort_loop(a, cut, last, depth, less);
        last = cut;
    }
}

/// `__unguarded_partition_pivot`: median-of-three pivot moved to `first`, then partition the rest.
fn unguarded_partition_pivot<T: Copy, F: Fn(T, T) -> bool>(
    a: &mut [T],
    first: usize,
    last: usize,
    less: &F,
) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(a, first, first + 1, mid, last - 1, less);
    unguarded_partition(a, first + 1, last, first, less)
}

/// `__move_median_to_first`: swaps the median of `a[x]`, `a[y]`, `a[z]` into `a[result]`.
fn move_median_to_first<T: Copy, F: Fn(T, T) -> bool>(
    a: &mut [T],
    result: usize,
    x: usize,
    y: usize,
    z: usize,
    less: &F,
) {
    let median = if less(a[x], a[y]) {
        if less(a[y], a[z]) {
            y
        } else if less(a[x], a[z]) {
            z
        } else {
            x
        }
    } else if less(a[x], a[z]) {
        x
    } else if less(a[y], a[z]) {
        z
    } else {
        y
    };
    a.swap(result, median);
}

/// `__unguarded_partition`: Hoare-style partition of `[first, last)` around `a[pivot]`.
fn unguarded_partition<T: Copy, F: Fn(T, T) -> bool>(
    a: &mut [T],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    less: &F,
) -> usize {
    loop {
        while less(a[first], a[pivot]) {
            first += 1;
        }
        last -= 1;
        while less(a[pivot], a[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        a.swap(first, last);
        first += 1;
    }
}

/// `__final_insertion_sort`.
fn final_insertion_sort<T: Copy, F: Fn(T, T) -> bool>(
    a: &mut [T],
    first: usize,
    last: usize,
    less: &F,
) {
    if last - first > SORT_THRESHOLD {
        insertion_sort(a, first, first + SORT_THRESHOLD, less);
        unguarded_insertion_sort(a, first + SORT_THRESHOLD, last, less);
    } else {
        insertion_sort(a, first, last, less);
    }
}

/// `__insertion_sort`.
fn insertion_sort<T: Copy, F: Fn(T, T) -> bool>(a: &mut [T], first: usize, last: usize, less: &F) {
    for i in first + 1..last {
        if less(a[i], a[first]) {
            let value = a[i];
            a.copy_within(first..i, first + 1);
            a[first] = value;
        } else {
            unguarded_linear_insert(a, i, less);
        }
    }
}

/// `__unguarded_insertion_sort`.
fn unguarded_insertion_sort<T: Copy, F: Fn(T, T) -> bool>(
    a: &mut [T],
    first: usize,
    last: usize,
    less: &F,
) {
    for i in first..last {
        unguarded_linear_insert(a, i, less);
    }
}

/// `__unguarded_linear_insert`: libstdc++ relies on an element at or before `last` that stops the
/// backward scan, so the scan never passes index 0.
fn unguarded_linear_insert<T: Copy, F: Fn(T, T) -> bool>(a: &mut [T], mut last: usize, less: &F) {
    let value = a[last];
    let mut next = last - 1;
    while less(value, a[next]) {
        a[last] = a[next];
        last = next;
        next -= 1;
    }
    a[last] = value;
}

/// `__partial_sort`: the heap fallback used when `__introsort_loop` runs out of depth.
fn partial_sort<T: Copy, F: Fn(T, T) -> bool>(
    a: &mut [T],
    first: usize,
    middle: usize,
    last: usize,
    less: &F,
) {
    heap_select(a, first, middle, last, less);
    sort_heap(a, first, middle, less);
}

/// `__heap_select`.
fn heap_select<T: Copy, F: Fn(T, T) -> bool>(
    a: &mut [T],
    first: usize,
    middle: usize,
    last: usize,
    less: &F,
) {
    make_heap(a, first, middle, less);
    for i in middle..last {
        if less(a[i], a[first]) {
            pop_heap(a, first, middle, i, less);
        }
    }
}

/// `__make_heap`.
fn make_heap<T: Copy, F: Fn(T, T) -> bool>(a: &mut [T], first: usize, last: usize, less: &F) {
    if last - first < 2 {
        return;
    }
    let len = last - first;
    let mut parent = (len - 2) / 2;
    loop {
        let value = a[first + parent];
        adjust_heap(a, first, parent, len, value, less);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

/// `__adjust_heap`: sifts `value` down from `hole`, then pushes it back up.
fn adjust_heap<T: Copy, F: Fn(T, T) -> bool>(
    a: &mut [T],
    first: usize,
    mut hole: usize,
    len: usize,
    value: T,
    less: &F,
) {
    let top = hole;
    let mut second = hole;
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if less(a[first + second], a[first + second - 1]) {
            second -= 1;
        }
        a[first + hole] = a[first + second];
        hole = second;
    }
    if len.is_multiple_of(2) && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        a[first + hole] = a[first + second - 1];
        hole = second - 1;
    }
    push_heap(a, first, hole, top, value, less);
}

/// `__push_heap`.
fn push_heap<T: Copy, F: Fn(T, T) -> bool>(
    a: &mut [T],
    first: usize,
    mut hole: usize,
    top: usize,
    value: T,
    less: &F,
) {
    let parent_of = |index: usize| if index == 0 { 0 } else { (index - 1) / 2 };
    let mut parent = parent_of(hole);
    while hole > top && less(a[first + parent], value) {
        a[first + hole] = a[first + parent];
        hole = parent;
        parent = parent_of(hole);
    }
    a[first + hole] = value;
}

/// `__pop_heap`: moves the top of the heap `[first, last)` to `a[result]`.
fn pop_heap<T: Copy, F: Fn(T, T) -> bool>(
    a: &mut [T],
    first: usize,
    last: usize,
    result: usize,
    less: &F,
) {
    let value = a[result];
    a[result] = a[first];
    adjust_heap(a, first, 0, last - first, value, less);
}

/// `__sort_heap`.
fn sort_heap<T: Copy, F: Fn(T, T) -> bool>(a: &mut [T], first: usize, mut last: usize, less: &F) {
    while last - first > 1 {
        last -= 1;
        pop_heap(a, first, last, last, less);
    }
}

/// Blender `do_dissolve`: dissolves dissolvable edges, longest first, while the result stays a
/// valid polygon set.
fn do_dissolve(fms: &mut FaceMergeState) {
    let mut dissolve_edges: Vec<usize> = fms
        .edge
        .iter()
        .enumerate()
        .filter(|(_, me)| me.dissolvable)
        .map(|(e, _)| e)
        .collect();
    // Blender's `std::ranges::sort` is unstable; `libstdcxx_sort_by` reproduces its tie order
    // for equal lengths on Linux (Blender 5.2.2).
    libstdcxx_sort_by(&mut dissolve_edges, &|a: usize, b: usize| {
        fms.edge[a].len_squared > fms.edge[b].len_squared
    });
    for me_index in dissolve_edges {
        let me = fms.edge[me_index];
        let (Some(left), Some(right)) = (me.left_face, me.right_face) else {
            continue;
        };
        if !dissolve_leaves_valid_bmesh(fms, me_index, left, right) {
            continue;
        }
        splice_faces(fms, me_index, left, right);
    }
}

/// Blender `merge_tris_for_face`: merges the triangles of one input face into output polygons.
/// Triangles that face opposite to the first one are merged in a second pass (Blender tries both
/// normal signs).
fn merge_tris_for_face(arena: &Arena, tris: &[Tri]) -> Vec<Poly> {
    let Some(first) = tris.first() else {
        return Vec::new();
    };
    if let [tri] = tris {
        return vec![poly_from_tri(tri)];
    }
    if let Some(poly) = same_side_quad(arena, tris) {
        return vec![poly];
    }
    let first_normal = tri_normal(arena, first);
    let mut out = Vec::new();
    for norm in [first_normal, -first_normal] {
        let mut fms = FaceMergeState::default();
        init_face_merge_state(&mut fms, arena, tris, norm);
        do_dissolve(&mut fms);
        for mf in &fms.face {
            if mf.merge_to.is_none() {
                out.push(Poly {
                    verts: mf.vert.clone(),
                    orig: mf.orig,
                    edge_orig: mf.edge.iter().map(|&e| fms.edge[e].orig).collect(),
                });
            }
        }
    }
    out
}

/// The two-triangle quad recovery, applied only when both triangles face the same way.
fn same_side_quad(arena: &Arena, tris: &[Tri]) -> Option<Poly> {
    let [tri1, tri2] = tris else {
        return None;
    };
    if tri_normal(arena, tri1).dot(tri_normal(arena, tri2)) <= 0.0 {
        return None;
    }
    recover_quad(tri1, tri2)
}

/// Blender `approx_in_line`: whether `b` lies on the ray `a -> c` direction, to within 1e-4 in
/// cosine.
fn approx_in_line(a: DVec3, b: DVec3, c: DVec3) -> bool {
    let cos_ang = normalize_blender(b - a).dot(normalize_blender(c - b));
    (cos_ang - 1.0).abs() < 1e-4
}

/// Blender `math::normalize` for doubles: a (near-)zero vector normalizes to zero.
fn normalize_blender(v: DVec3) -> DVec3 {
    let len_sq = v.length_squared();
    if len_sq > 1.0e-70 {
        v / len_sq.sqrt()
    } else {
        DVec3::ZERO
    }
}

/// Blender `find_dissolve_verts`: the non-input vertices of valence 2 whose two neighbours and
/// themselves are in line.
fn find_dissolve_verts(arena: &Arena, polys: &[Poly]) -> BTreeSet<u32> {
    // Per used vertex: (still a candidate, first-seen neighbours as (next, prev)).
    let mut state: BTreeMap<u32, (bool, Option<(u32, u32)>)> = BTreeMap::new();
    for poly in polys {
        let n = poly.verts.len();
        for (i, &v) in poly.verts.iter().enumerate() {
            let entry = state
                .entry(v)
                .or_insert((arena.vert(v).orig == NO_INDEX, None));
            if !entry.0 {
                continue;
            }
            let n1 = poly.verts[(i + 1) % n];
            let n2 = poly.verts[(i + n - 1) % n];
            match entry.1 {
                None => entry.1 = Some((n1, n2)),
                Some((f1, f2)) => {
                    // Seen in another face with different neighbours: not valence 2.
                    if !((n1 == f2 && n2 == f1) || (n1 == f1 && n2 == f2)) {
                        entry.0 = false;
                    }
                }
            }
        }
    }
    let mut dissolved = BTreeSet::new();
    for (&v, &(candidate, nbrs)) in &state {
        let Some((n1, n2)) = nbrs else {
            continue;
        };
        if candidate && approx_in_line(arena.vert(n1).co_f, arena.vert(v).co_f, arena.vert(n2).co_f)
        {
            dissolved.insert(v);
        }
    }
    dissolved
}

/// Blender `dissolve_verts`: drops the dissolved vertices from every polygon. A polygon left with
/// fewer than three vertices is removed. Each kept ring edge keeps the `edge_orig` of its start
/// vertex, as Blender's `erase_face_positions` does.
fn erase_dissolved_verts(polys: Vec<Poly>, dissolved: &BTreeSet<u32>) -> Vec<Poly> {
    let mut out = Vec::with_capacity(polys.len());
    for poly in polys {
        if !poly.verts.iter().any(|v| dissolved.contains(v)) {
            out.push(poly);
            continue;
        }
        let kept: Vec<usize> = poly
            .verts
            .iter()
            .enumerate()
            .filter(|&(_, v)| !dissolved.contains(v))
            .map(|(i, _)| i)
            .collect();
        if kept.len() < 3 {
            continue;
        }
        out.push(Poly {
            verts: kept.iter().map(|&i| poly.verts[i]).collect(),
            orig: poly.orig,
            edge_orig: kept.iter().map(|&i| poly.edge_orig[i]).collect(),
        });
    }
    out
}

/// Blender `find_dissolve_verts` followed by `dissolve_verts`.
fn dissolve_vertices(arena: &Arena, polys: Vec<Poly>) -> Vec<Poly> {
    let dissolved = find_dissolve_verts(arena, &polys);
    if dissolved.is_empty() {
        return polys;
    }
    erase_dissolved_verts(polys, &dissolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vert(arena: &mut Arena, x: f64, y: f64, orig: i32) -> u32 {
        arena.add_or_find_dvec(DVec3::new(x, y, 0.0), orig)
    }

    #[test]
    fn quad_split_by_diagonal_restores_input_quad() {
        let mut arena = Arena::new();
        let a = vert(&mut arena, 0.0, 0.0, 0);
        let b = vert(&mut arena, 2.0, 0.0, 1);
        let c = vert(&mut arena, 2.0, 2.0, 2);
        let d = vert(&mut arena, 0.0, 2.0, 3);
        let tris = [
            Tri::new([a, b, c], 0, [0, 1, NO_INDEX]),
            Tri::new([a, c, d], 0, [NO_INDEX, 2, 3]),
        ];
        assert_eq!(
            polymesh_from_tris(&arena, &tris),
            vec![Poly {
                verts: vec![a, b, c, d],
                orig: 0,
                edge_orig: vec![0, 1, 2, 3],
            }]
        );
    }

    #[test]
    fn fan_around_off_line_centre_keeps_centre() {
        let mut arena = Arena::new();
        let a = vert(&mut arena, 0.0, 0.0, 0);
        let b = vert(&mut arena, 2.0, 0.0, 1);
        let c = vert(&mut arena, 2.0, 2.0, 2);
        let d = vert(&mut arena, 0.0, 2.0, 3);
        let o = vert(&mut arena, 1.0, 1.0, NO_INDEX);
        let tris = [
            Tri::new([o, a, b], 0, [NO_INDEX, 0, NO_INDEX]),
            Tri::new([o, b, c], 0, [NO_INDEX, 1, NO_INDEX]),
            Tri::new([o, c, d], 0, [NO_INDEX, 2, NO_INDEX]),
            Tri::new([o, d, a], 0, [NO_INDEX, 3, NO_INDEX]),
        ];
        // Blender's merge order: the spokes O-A, O-B, O-C, O-D dissolve (equal lengths, creation
        // order). O-C and O-D stay: removing O-D would repeat C in the merged face. The centre has
        // neighbours D and C, which are not collinear with it, so it survives.
        assert_eq!(
            polymesh_from_tris(&arena, &tris),
            vec![
                Poly {
                    verts: vec![o, c, d],
                    orig: 0,
                    edge_orig: vec![NO_INDEX, 2, NO_INDEX],
                },
                Poly {
                    verts: vec![o, d, a, b, c],
                    orig: 0,
                    edge_orig: vec![NO_INDEX, 3, 0, 1, NO_INDEX],
                },
            ]
        );
    }

    #[test]
    fn cut_vertex_on_straight_edge_is_dissolved() {
        let mut arena = Arena::new();
        let a = vert(&mut arena, 0.0, 0.0, 0);
        let b = vert(&mut arena, 2.0, 0.0, 1);
        let c = vert(&mut arena, 2.0, 2.0, 2);
        let d = vert(&mut arena, 0.0, 2.0, 3);
        let m = vert(&mut arena, 1.0, 0.0, NO_INDEX);
        let tris = [
            Tri::new([a, m, c], 0, [0, NO_INDEX, NO_INDEX]),
            Tri::new([a, c, d], 0, [NO_INDEX, 2, 3]),
            Tri::new([m, b, c], 0, [0, 1, NO_INDEX]),
        ];
        assert_eq!(
            polymesh_from_tris(&arena, &tris),
            vec![Poly {
                verts: vec![b, c, d, a],
                orig: 0,
                edge_orig: vec![1, 2, 3, 0],
            }]
        );
    }

    #[test]
    fn faces_of_different_origin_never_merge() {
        let mut arena = Arena::new();
        let a = vert(&mut arena, 0.0, 0.0, 0);
        let b = vert(&mut arena, 2.0, 0.0, 1);
        let c = vert(&mut arena, 2.0, 2.0, 2);
        let d = vert(&mut arena, 0.0, 2.0, 3);
        // The shared diagonal A-C is dissolvable, but the two triangles come from different
        // input faces, so they stay separate. Output is ordered by input face.
        let tris = [
            Tri::new([a, c, d], 1, [NO_INDEX, 2, 3]),
            Tri::new([a, b, c], 0, [0, 1, NO_INDEX]),
        ];
        assert_eq!(
            polymesh_from_tris(&arena, &tris),
            vec![
                Poly {
                    verts: vec![a, b, c],
                    orig: 0,
                    edge_orig: vec![0, 1, NO_INDEX],
                },
                Poly {
                    verts: vec![a, c, d],
                    orig: 1,
                    edge_orig: vec![NO_INDEX, 2, 3],
                },
            ]
        );
    }

    #[test]
    fn edge_origins_follow_ring_edges() {
        let mut arena = Arena::new();
        let a = vert(&mut arena, 0.0, 0.0, 0);
        let b = vert(&mut arena, 2.0, 0.0, 1);
        let c = vert(&mut arena, 2.0, 2.0, 2);
        let d = vert(&mut arena, 0.0, 2.0, 3);
        let m = vert(&mut arena, 1.0, 0.0, NO_INDEX);
        // Same triangulation as `cut_vertex_on_straight_edge_is_dissolved`, but the cut edge
        // M-C is not dissolvable (intersection edge), so M stays on the boundary of two faces.
        let tris = [
            Tri {
                v: [a, m, c],
                orig: 0,
                edge_orig: [0, NO_INDEX, NO_INDEX],
                is_intersect: [false, true, false],
            },
            Tri::new([a, c, d], 0, [NO_INDEX, 2, 3]),
            Tri {
                v: [m, b, c],
                orig: 0,
                edge_orig: [0, 1, NO_INDEX],
                is_intersect: [false, false, true],
            },
        ];
        let polys = polymesh_from_tris(&arena, &tris);
        assert_eq!(
            polys,
            vec![
                Poly {
                    verts: vec![a, m, c, d],
                    orig: 0,
                    edge_orig: vec![0, NO_INDEX, 2, 3],
                },
                Poly {
                    verts: vec![m, b, c],
                    orig: 0,
                    edge_orig: vec![0, 1, NO_INDEX],
                },
            ]
        );
        assert!(!polys[0].is_new_edge(0));
        assert!(polys[0].is_new_edge(1));
        assert!(!polys[0].is_new_edge(2));
        assert!(polys[1].is_new_edge(2));
    }
}
