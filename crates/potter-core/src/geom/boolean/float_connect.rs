//! Blender 5.2.2 `BM_face_split_edgenet_connect_islands` for the FLOAT solver.
//!
//! Port of `source/blender/bmesh/intern/bmesh_polygon_edgenet.cc` with `use_partial_connect = false`,
//! called from `face_edges_split` in `source/blender/bmesh/tools/bmesh_intersect.cc` before
//! `BM_face_split_edgenet`. The cut-edge net of a face is split into islands; every island after the
//! first gets a connection from its lowest vertex and one from its highest vertex, each to the nearest
//! visible vertex along the face's sort axis, in the face's 2D frame.
//!
//! Arithmetic is f32 in Blender's operand order, with the same 2D frame (`ortho_basis_v3v3_v3` of
//! `f->no`), the same origin (`BM_FACE_FIRST_LOOP(f)->v`), and the same endpoint bias (`1e-4`).
//! Departures from the C code, neither of which changes the recorded Blender results:
//! - The BVH ray casts are linear scans over the same edge list with the same predicates. Blender's
//!   BVH breaks exact distance ties by traversal order; this port breaks them by edge order.
//! - Islands come from union-find rather than the disk-cycle DFS. Island and vertex order only break
//!   exact projection ties, which here take the first vertex in edge order.
//!
//! Blender's arm64 FMA contraction is not reproduced; the FLOAT path rounds each operation separately.

use std::cmp::Ordering;
use std::collections::HashMap;

type P2 = [f32; 2];
type P3 = [f32; 3];

/// Blender `BVH_RAYCAST_DIST_MAX`.
const RAYCAST_DIST_MAX: f32 = f32::MAX / 2.0;
/// Blender `edge_isect_verts_point_2d` endpoint bias.
const ENDPOINT_BIAS: f32 = 1.0e-4;

/// Returns the connection edges Blender creates for one face, in creation order, as `(origin, end)`.
///
/// - `reference`: `BM_FACE_FIRST_LOOP(f)->v`, the vertex the 2D frame is relative to.
/// - `normal`: `f->no`.
/// - `loop_edges`: the face's loop edges from `reference` onwards, as stored on the `BMesh` edges.
/// - `net`: the cut-edge net in Blender's `edge_net_init` list order, as stored on the `BMesh` edges.
/// - `positions`: coordinates indexed by vertex id.
///
/// A returned pair may repeat an earlier pair or an input edge: Blender's `BM_CREATE_NO_DOUBLE` hands
/// back the existing edge, with that edge's stored orientation, which is still listed. Only new edges
/// carry `(origin, end)` orientation, and nothing downstream reads it. The caller appends the result after `net`.
/// Returns nothing when the net and loop form a single island.
pub(super) fn connect_islands(
    reference: usize,
    normal: P3,
    loop_edges: &[(usize, usize)],
    net: &[(usize, usize)],
    positions: &[P3],
) -> Vec<(usize, usize)> {
    let mut vert_ids: Vec<usize> = Vec::new();
    let mut local: HashMap<usize, usize> = HashMap::new();
    // Blender `edge_arr`: the cut net, then the face loop.
    let edges: Vec<(usize, usize)> = net
        .iter()
        .chain(loop_edges)
        .map(|&(a, b)| {
            (
                intern(a, &mut vert_ids, &mut local),
                intern(b, &mut vert_ids, &mut local),
            )
        })
        .collect();
    let n = vert_ids.len();
    let co: Vec<P3> = vert_ids.iter().map(|&v| positions[v]).collect();

    let islands = edge_islands(n, &edges);
    if islands.len() < 2 {
        return Vec::new();
    }

    // `axis_dominant_v3_to_m3(f->no)`: rows n1 and n2 of the frame; projection is a dot product.
    let (n1, n2) = ortho_basis(normal);
    // Island spans use raw coordinates, as in Blender's group setup.
    let raw: Vec<P2> = co.iter().map(|&c| [dot3(n1, c), dot3(n2, c)]).collect();
    let mut spans: Vec<Island> = islands
        .into_iter()
        .map(|vertices| Island::new(vertices, &raw))
        .collect();
    spans.sort_by(|a, b| {
        axis_pt_cmp(a.min_axis, b.min_axis).then_with(|| axis_pt_cmp(a.max_axis, b.max_axis))
    });
    let mut group = vec![0; n];
    for (rank, island) in spans.iter().enumerate() {
        for &v in &island.vertices {
            group[v] = rank;
        }
    }

    // Blender converts to 2D relative to `f_co_ref`, then uses that for every ray test.
    let ref_co = positions[reference];
    let rel: Vec<P2> = co
        .iter()
        .map(|&c| {
            let d = sub3(c, ref_co);
            [dot3(n1, d), dot3(n2, d)]
        })
        .collect();

    let mut state = Islands {
        edges,
        rel,
        group,
        valid: vec![true; n],
        added: Vec::new(),
    };
    let mut has_prev = vec![false; spans.len()];
    for g in 1..spans.len() {
        if !has_prev[g] {
            let origin = spans[g].min;
            if let Some(target) = state.find_connection(origin, false, g) {
                state.added.push((origin, target));
            }
        }
        let origin = spans[g].max;
        if let Some(target) = state.find_connection(origin, true, g) {
            state.added.push((origin, target));
            has_prev[state.group[target]] = true;
        }
    }
    state
        .added
        .iter()
        .map(|&(a, b)| (vert_ids[a], vert_ids[b]))
        .collect()
}

fn intern(v: usize, vert_ids: &mut Vec<usize>, local: &mut HashMap<usize, usize>) -> usize {
    *local.entry(v).or_insert_with(|| {
        vert_ids.push(v);
        vert_ids.len() - 1
    })
}

/// Blender `EdgeGroupIsland` before sorting: the member vertices and their projection span.
struct Island {
    vertices: Vec<usize>,
    /// Lowest vertex by raw projection (`vert_span.min`).
    min: usize,
    /// Highest vertex by raw projection (`vert_span.max`).
    max: usize,
    min_axis: P2,
    max_axis: P2,
}

impl Island {
    fn new(vertices: Vec<usize>, raw: &[P2]) -> Self {
        let mut min = vertices[0];
        let mut max = vertices[0];
        for &v in &vertices[1..] {
            if axis_pt_cmp(raw[v], raw[min]) == Ordering::Less {
                min = v;
            }
            if axis_pt_cmp(raw[v], raw[max]) == Ordering::Greater {
                max = v;
            }
        }
        Self {
            min,
            max,
            min_axis: raw[min],
            max_axis: raw[max],
            vertices,
        }
    }
}

/// Connected components of the edge graph, in order of first vertex.
fn edge_islands(vertex_count: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..vertex_count).collect();
    for &(a, b) in edges {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }
    let mut slot = vec![usize::MAX; vertex_count];
    let mut islands: Vec<Vec<usize>> = Vec::new();
    for v in 0..vertex_count {
        let root = find(&mut parent, v);
        if slot[root] == usize::MAX {
            slot[root] = islands.len();
            islands.push(Vec::new());
        }
        islands[slot[root]].push(v);
    }
    islands
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Blender `Islands` state of one `BM_face_split_edgenet_connect_islands` call.
struct Islands {
    /// Local vertex ids; Blender `edge_arr` order.
    edges: Vec<(usize, usize)>,
    /// 2D coordinates relative to the face reference, by local vertex id.
    rel: Vec<P2>,
    /// Island rank by local vertex id.
    group: Vec<usize>,
    /// Blender `VERT_IS_VALID`: cleared while a vertex is queued in one connection search.
    valid: Vec<bool>,
    /// Blender `edge_arr_new`: connections created so far, duplicates included.
    added: Vec<(usize, usize)>,
}

impl Islands {
    /// Blender `bm_face_split_edgenet_find_connection`.
    fn find_connection(&mut self, origin: usize, sign: bool, island: usize) -> Option<usize> {
        let mut e_hit = self.ray_hit(origin, sign, island)?;
        let mut v_other: Option<usize> = None;
        let mut v_other_fallback: Option<usize>;
        let mut search: Vec<usize> = Vec::new();
        let mut blacklist: Vec<usize> = Vec::new();
        let o = self.rel[origin];
        loop {
            // The closer endpoint is popped first.
            let (v1, v2) = e_hit;
            let pair = if len_sq2(o, self.rel[v1]) > len_sq2(o, self.rel[v2]) {
                [v1, v2]
            } else {
                [v2, v1]
            };
            for v in pair {
                let x = self.rel[v][0];
                let past = if sign { x > o[0] } else { x < o[0] };
                if self.valid[v] && past {
                    search.push(v);
                    blacklist.push(v);
                    self.valid[v] = false;
                }
            }
            v_other_fallback = v_other;
            v_other = search.pop();
            let Some(candidate) = v_other else { break };
            match self.vert_hit(origin, candidate, island) {
                Some(blocker) => e_hit = blocker,
                None => break,
            }
        }
        if v_other.is_none() {
            v_other = v_other_fallback;
        }
        for v in blacklist {
            self.valid[v] = true;
        }
        v_other
    }

    /// Blender `test_edges_isect_2d_ray`: the edge hit by a ray along +x or -x from `origin`.
    fn ray_hit(&self, origin: usize, sign: bool, island: usize) -> Option<(usize, usize)> {
        let o = self.rel[origin];
        let dir: P2 = [if sign { 1.0 } else { -1.0 }, 0.0];
        let mut best = RAYCAST_DIST_MAX;
        let mut hit = None;
        for &e in &self.edges {
            let Some(dist) = isect_ray_seg(o, dir, self.rel[e.0], self.rel[e.1]) else {
                continue;
            };
            if dist < best
                && dist > 0.0
                && e.0 != origin
                && e.1 != origin
                && self.group[e.0] != island
            {
                best = dist;
                hit = Some(e);
            }
        }
        // Blender checks the new edges only when the BVH found a hit.
        hit?;
        for &e in &self.added {
            if let Some(dist) = isect_ray_seg(o, dir, self.rel[e.0], self.rel[e.1])
                && e.0 != origin
                && e.1 != origin
                && dist < best
            {
                best = dist;
                hit = Some(e);
            }
        }
        hit
    }

    /// Blender `test_edges_isect_2d_vert`: the edge that blocks the segment `origin -> other`.
    fn vert_hit(&self, origin: usize, other: usize, island: usize) -> Option<(usize, usize)> {
        let a = self.rel[origin];
        let b = self.rel[other];
        let mut dir = sub2(b, a);
        let dist_orig = normalize2(&mut dir);
        let mut best = dist_orig;
        let mut hit = None;
        for &e in &self.edges {
            let Some(co) = self.vert_edge_isect(e, origin, other) else {
                continue;
            };
            let dist_new = dist_orig * line_point_factor(co, a, b);
            if dist_new < best && dist_new > 0.0 && self.group[e.0] != island {
                best = dist_new;
                hit = Some(e);
            }
        }
        if hit.is_some() {
            return hit;
        }
        let mut t_best = 1.0;
        for &e in &self.added {
            if let Some(co) = self.vert_edge_isect(e, origin, other) {
                let t = line_point_factor(co, a, b);
                if t < t_best {
                    t_best = t;
                    hit = Some(e);
                }
            }
        }
        hit
    }

    /// Blender `edge_isect_verts_point_2d`.
    fn vert_edge_isect(&self, edge: (usize, usize), a: usize, b: usize) -> Option<P2> {
        if edge.0 == a || edge.1 == a || edge.0 == b || edge.1 == b {
            return None;
        }
        isect_seg_seg_point(
            self.rel[a],
            self.rel[b],
            self.rel[edge.0],
            self.rel[edge.1],
            ENDPOINT_BIAS,
        )
    }
}

/// Blender `axis_pt_cmp`: lexicographic on (x, y).
fn axis_pt_cmp(a: P2, b: P2) -> Ordering {
    if a[0] < b[0] {
        Ordering::Less
    } else if a[0] > b[0] {
        Ordering::Greater
    } else if a[1] < b[1] {
        Ordering::Less
    } else if a[1] > b[1] {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// Blender `ortho_basis_v3v3_v3`: the two tangents of `n`, in the order used by `axis_dominant_v3_to_m3`.
fn ortho_basis(n: P3) -> (P3, P3) {
    let f = n[0] * n[0] + n[1] * n[1];
    if f > f32::EPSILON {
        let d = 1.0 / f.sqrt();
        let n1 = [n[1] * d, -n[0] * d, 0.0];
        let n2 = [-n[2] * n1[1], n[2] * n1[0], n[0] * n1[1] - n[1] * n1[0]];
        (n1, n2)
    } else {
        let n1 = [if n[2] < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0];
        (n1, [0.0, 1.0, 0.0])
    }
}

fn dot3(a: P3, b: P3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub3(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn sub2(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot2(a: P2, b: P2) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross2(a: P2, b: P2) -> f32 {
    a[0] * b[1] - a[1] * b[0]
}

/// Blender `len_squared_v2v2(a, b)`.
fn len_sq2(a: P2, b: P2) -> f32 {
    let d = sub2(b, a);
    dot2(d, d)
}

/// Blender `normalize_v2`: scales `v` to unit length and returns the length (zero for tiny vectors).
fn normalize2(v: &mut P2) -> f32 {
    let d = dot2(*v, *v);
    if d > 1.0e-35 {
        let len = d.sqrt();
        let scale = 1.0 / len;
        *v = [v[0] * scale, v[1] * scale];
        len
    } else {
        *v = [0.0, 0.0];
        0.0
    }
}

/// Blender `line_point_factor_v2` with zero epsilon and fallback.
fn line_point_factor(p: P2, l1: P2, l2: P2) -> f32 {
    let u = sub2(l2, l1);
    let h = sub2(p, l1);
    let len = dot2(u, u);
    if len > 0.0 { dot2(u, h) / len } else { 0.0 }
}

/// Blender `isect_ray_seg_v2` with a unit-length ray direction: the distance along the ray.
fn isect_ray_seg(origin: P2, dir: P2, v0: P2, v1: P2) -> Option<f32> {
    let v0l = sub2(v0, origin);
    let v1l = sub2(v1, origin);
    let s10 = sub2(v1l, v0l);
    let det = cross2(dir, s10);
    if det == 0.0 {
        return None;
    }
    let v = cross2(v0l, v1l);
    let p = [(dir[0] * v) / det, (dir[1] * v) / det];
    let t = dot2(p, dir) / dot2(dir, dir);
    if !(0.0..).contains(&t) {
        return None;
    }
    let h = sub2(v1l, p);
    let u = dot2(s10, h) / dot2(s10, s10);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    Some(t)
}

/// Blender `isect_seg_seg_v2_point_ex`: the intersection point of two segments with endpoint bias.
#[expect(
    clippy::float_cmp,
    reason = "exact collinear-overlap equality mirrors Blender's bit-exact check; tolerances would change results"
)]
fn isect_seg_seg_point(v0: P2, v1: P2, v2: P2, v3: P2, bias: f32) -> Option<P2> {
    let endpoint_min = -bias;
    let endpoint_max = bias + 1.0;
    let in_range = |x: f32| (endpoint_min..=endpoint_max).contains(&x);
    let s10 = sub2(v1, v0);
    let s32 = sub2(v3, v2);
    let s30 = sub2(v3, v0);
    let d = cross2(s10, s32);
    if d != 0.0 {
        let u = cross2(s30, s32) / d;
        let v = cross2(s10, s30) / d;
        if in_range(u) && in_range(v) {
            let vi = [v0[0] + s10[0] * u, v0[1] + s10[1] * u];
            let s_vi_v2 = sub2(vi, v2);
            if in_range(dot2(s32, s_vi_v2) / dot2(s32, s32)) {
                return Some(vi);
            }
        }
        return None;
    }
    if cross2(s10, s30) != 0.0 || cross2(s32, s30) != 0.0 {
        return None;
    }

    // Collinear: use the non-degenerate segment as the basis.
    let (mut a0, mut a1, mut b0, mut b1) = (v0, v1, v2, v3);
    let mut basis = s10;
    let mut to_b = s30;
    if a0 == a1 {
        let eps = 1.0e-6_f32;
        if len_sq2(b0, b1) > eps * eps {
            std::mem::swap(&mut a0, &mut b0);
            std::mem::swap(&mut a1, &mut b1);
            basis = sub2(a1, a0);
            to_b = sub2(b1, a0);
        } else {
            return (a0 == b0).then_some(a0);
        }
    }

    let s20 = sub2(b0, a0);
    let mut u_a = dot2(s20, basis) / dot2(basis, basis);
    let mut u_b = dot2(to_b, basis) / dot2(basis, basis);
    if u_a > u_b {
        std::mem::swap(&mut u_a, &mut u_b);
    }
    if u_a > endpoint_max || u_b < endpoint_min {
        return None;
    }
    let lo = if 0.0 > u_a { 0.0 } else { u_a };
    let hi = if 1.0 < u_b { 1.0 } else { u_b };
    if lo == hi {
        return Some([a0[0] + basis[0] * lo, a0[1] + basis[1] * lo]);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::connect_islands;
    use serde_json::Value;

    const FIXTURE: &str = include_str!(
        "../../../../../tests/fixtures/boolean_parity/connect_islands/coplanar_float.json"
    );

    fn index(value: &Value) -> usize {
        value.as_u64().map_or(usize::MAX, |v| v as usize)
    }

    fn pairs(value: &Value) -> Vec<(usize, usize)> {
        value.as_array().map_or_else(Vec::new, |items| {
            items
                .iter()
                .map(|pair| (index(&pair[0]), index(&pair[1])))
                .collect()
        })
    }

    fn bits3(value: &Value) -> [f32; 3] {
        let bits = |i: usize| value[i].as_u64().map_or(0, |b| b as u32);
        [bits(0), bits(1), bits(2)].map(f32::from_bits)
    }

    #[test]
    fn connect_islands_matches_blender_records_exactly() {
        let Ok(root) = serde_json::from_str::<Value>(FIXTURE) else {
            panic!("connect_islands fixture is valid JSON");
        };
        let Some(cases) = root["cases"].as_object() else {
            panic!("connect_islands fixture has a cases object");
        };
        let mut connected = 0;
        for (name, records) in cases {
            let Some(records) = records.as_array() else {
                panic!("{name}: records must be an array");
            };
            for (number, record) in records.iter().enumerate() {
                let positions: Vec<[f32; 3]> = record["positions"]
                    .as_array()
                    .map(|rows| rows.iter().map(bits3).collect())
                    .unwrap_or_default();
                let actual = connect_islands(
                    index(&record["ring"][0]),
                    bits3(&record["normal"]),
                    &pairs(&record["loop_edges"]),
                    &pairs(&record["net_init"]),
                    &positions,
                );
                let expected = pairs(&record["new_edges"]);
                // Blender returns pre-existing edges in their stored orientation, so compare unordered pairs.
                let unordered = |edges: &[(usize, usize)]| -> Vec<(usize, usize)> {
                    edges.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect()
                };
                assert_eq!(unordered(&actual), unordered(&expected), "{name}[{number}]");
                connected += usize::from(!expected.is_empty());
            }
        }
        assert!(connected > 0, "fixture exercises at least one connection");
    }

    #[test]
    fn single_island_creates_no_connections() {
        let square = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let loop_edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
        let connects = connect_islands(0, [0.0, 0.0, 1.0], &loop_edges, &[], &square);
        assert_eq!(connects, Vec::<(usize, usize)>::new());
    }
}
