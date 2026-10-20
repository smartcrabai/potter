//! Blender's ear-clipping polyfill, the kd-tree variant (`BLI_polyfill_calc_arena`,
//! `blenlib/intern/polyfill_2d.cc`, `coords_sign == 1`), the one routine behind every polygon
//! triangulation in Potter: Mesh tessellation (`mesh_tessellate.cc:116`, `BLI_polyfill_calc_arena`),
//! Boolean EXACT and FLOAT n-gon triangulation (`mesh_intersect.cc:1953`, `BLI_polyfill_calc`), and
//! Decimate's COLLAPSE n-gon fill.
//!
//! Blender 5.2.2 compiles that file with `USE_CLIP_EVEN`, `USE_CONVEX_SKIP` and `USE_CLIP_SWEEP`
//! defined unconditionally (`polyfill_2d.cc:54-57`). `USE_CONVEX_SKIP` implies `USE_KDTREE`
//! (`:60-62`), which implies `USE_KDTREE_INDEX_CACHE` (`:66`) and `USE_PRECOMPUTED_ISECT` (`:71`).
//! The kd-tree is built only when a corner is concave (`polyfill_2d.cc:1050-1056`), and the index
//! cache keeps the last blocking vertex. Only those paths are kept.
//!
//! Float operation order and tie-breaking follow the original, since the triangle order feeds the
//! Mesh tessellation and Decimate's collapse order. On degenerate rings (duplicate points) the
//! brute-force candidate search this replaced disagreed with Blender's `loop_triangles`; this
//! variant agrees on all clockwise rings with five or more corners (`tests/fixtures/polyfill`).

/// Stands in for Blender's `UINT_MAX` / `KDNODE_UNSET` sentinels.
const NONE: usize = usize::MAX;

type Sign = i8;
/// Blender's `CONCAVE` / `TANGENTIAL` / `CONVEX` enum.
const CONCAVE: Sign = -1;
const TANGENTIAL: Sign = 0;
const CONVEX: Sign = 1;

/// Blender's `signum_enum`.
fn signum_enum(a: f32) -> Sign {
    if a > 0.0 {
        CONVEX
    } else if a == 0.0 {
        TANGENTIAL
    } else {
        CONCAVE
    }
}

/// Blender's `area_tri_signed_v2_alt_2x`.
fn area_tri_signed_alt_2x(v1: [f32; 2], v2: [f32; 2], v3: [f32; 2]) -> f32 {
    let d2 = [v2[0] - v1[0], v2[1] - v1[1]];
    let d3 = [v3[0] - v1[0], v3[1] - v1[1]];
    (d2[0] * d3[1]) - (d3[0] * d2[1])
}

/// Blender's `span_tri_v2_sign`.
fn span_tri_sign(v1: [f32; 2], v2: [f32; 2], v3: [f32; 2]) -> Sign {
    signum_enum(area_tri_signed_alt_2x(v3, v2, v1))
}

/// `std::min` over three values, keeping the first on ties.
fn min3(a: f32, b: f32, c: f32) -> f32 {
    let mut m = a;
    if b < m {
        m = b;
    }
    if c < m {
        m = c;
    }
    m
}

/// `std::max` over three values, keeping the first on ties.
fn max3(a: f32, b: f32, c: f32) -> f32 {
    let mut m = a;
    if m < b {
        m = b;
    }
    if m < c {
        m = c;
    }
    m
}

/// Blender's `TriIsectPrecomputed`: edge vectors and per-edge constants for point-in-triangle tests.
struct TriIsect {
    edge: [[f32; 2]; 3],
    c: [f32; 3],
}

impl TriIsect {
    /// Blender's `tri_isect_precomputed_init`.
    fn new(vs: [[f32; 2]; 3]) -> Self {
        let edge = [
            [vs[1][0] - vs[0][0], vs[1][1] - vs[0][1]],
            [vs[2][0] - vs[1][0], vs[2][1] - vs[1][1]],
            [vs[0][0] - vs[2][0], vs[0][1] - vs[2][1]],
        ];
        let c = [0, 1, 2].map(|i| (edge[i][0] * vs[i][1]) - (vs[i][0] * edge[i][1]));
        Self { edge, c }
    }

    /// Blender's `tri_isect_precomputed_test`: inside or on the edge.
    fn contains(&self, co: [f32; 2]) -> bool {
        let e = &self.edge;
        ((e[1][1] * co[0]) - (e[1][0] * co[1]) + self.c[1] >= 0.0)
            && ((e[2][1] * co[0]) - (e[2][0] * co[1]) + self.c[2] >= 0.0)
            && ((e[0][1] * co[0]) - (e[0][0] * co[1]) + self.c[0] >= 0.0)
    }
}

/// Blender's `Range` (`KDRange2D`).
#[derive(Clone, Copy)]
struct Range {
    min: f32,
    max: f32,
}

/// Blender's `KDTreeNode2D`; `neg`/`pos`/`parent` are node slots, `index` is the vertex.
#[derive(Clone, Copy)]
struct KdNode {
    neg: usize,
    pos: usize,
    index: usize,
    axis: usize,
    removed: bool,
    parent: usize,
}

/// Blender's `KDTree2D`: a spatial index over the concave vertices, used by the ear test.
struct KdTree {
    nodes: Vec<KdNode>,
    root: usize,
    /// Vertex index to node slot, `NONE` once the vertex left the tree.
    nodes_map: Vec<usize>,
}

/// Blender's `balance_recursive` (`kdtree2d_balance_recursive`), on a subslice whose nodes start at
/// global slot `ofs`.
fn balance(nodes: &mut [KdNode], axis: usize, coords: &[[f32; 2]], ofs: usize) -> usize {
    let node_num = nodes.len();
    if node_num == 0 {
        return NONE;
    }
    if node_num == 1 {
        return ofs;
    }

    // Quick-sort style partition around the median; signed indices keep Blender's `neg - 1` exact.
    let median = node_num / 2;
    let median_i = median as isize;
    let mut neg: isize = 0;
    let mut pos: isize = node_num as isize - 1;
    while pos > neg {
        let co = coords[nodes[pos as usize].index][axis];
        let mut i = neg - 1;
        let mut j = pos;
        loop {
            i += 1;
            while coords[nodes[i as usize].index][axis] < co {
                i += 1;
            }
            j -= 1;
            while coords[nodes[j as usize].index][axis] > co && j > neg {
                j -= 1;
            }
            if i >= j {
                break;
            }
            nodes.swap(i as usize, j as usize);
        }
        nodes.swap(i as usize, pos as usize);
        if i >= median_i {
            pos = i - 1;
        }
        if i <= median_i {
            neg = i + 1;
        }
    }

    nodes[median].axis = axis;
    let neg_root = balance(&mut nodes[..median], axis ^ 1, coords, ofs);
    let pos_root = balance(&mut nodes[median + 1..], axis ^ 1, coords, median + 1 + ofs);
    nodes[median].neg = neg_root;
    nodes[median].pos = pos_root;
    median + ofs
}

/// Blender's `BLI_polyfill_calc_arena` `kdtree2d_*` stack: `build` mirrors `kdtree2d_new`,
/// `kdtree2d_init`, `kdtree2d_balance` and `kdtree2d_init_mapping`.
impl KdTree {
    fn build(coords: &[[f32; 2]], pis: &[PolyIndex]) -> Self {
        let mut nodes: Vec<KdNode> = pis
            .iter()
            .filter(|pi| pi.sign != CONVEX)
            .map(|pi| KdNode {
                neg: NONE,
                pos: NONE,
                index: pi.index,
                axis: 0,
                removed: false,
                parent: NONE,
            })
            .collect();
        let root = balance(&mut nodes, 0, coords, 0);

        let mut nodes_map = vec![NONE; coords.len()];
        for i in 0..nodes.len() {
            let node = nodes[i];
            if node.neg != NONE {
                nodes[node.neg].parent = i;
            }
            if node.pos != NONE {
                nodes[node.pos].parent = i;
            }
            nodes_map[node.index] = i;
        }
        nodes[root].parent = NONE;

        Self {
            nodes,
            root,
            nodes_map,
        }
    }

    /// Blender's `kdtree2d_node_remove`: unlinks `index`, collapsing childless and single-child
    /// removed ancestors so the tree path stays short.
    fn remove(&mut self, index: usize) {
        let mut node = self.nodes_map[index];
        if node == NONE {
            return;
        }
        self.nodes_map[index] = NONE;
        self.nodes[node].removed = true;

        while self.nodes[node].parent != NONE {
            let cur = self.nodes[node];
            let node_child = if cur.neg == NONE {
                cur.pos
            } else if cur.pos == NONE {
                cur.neg
            } else {
                break;
            };
            let parent = cur.parent;
            let node_parent = &mut self.nodes[parent];
            if node_parent.neg == node {
                node_parent.neg = node_child;
            } else {
                node_parent.pos = node_child;
            }
            if node_child != NONE {
                self.nodes[node_child].parent = parent;
            }
            if !self.nodes[parent].removed {
                break;
            }
            node = parent;
        }
    }

    /// Blender's `kdtree2d_isect_tri_single`: whether `test` is still in the tree and inside the
    /// triangle `ind`.
    fn isect_single(&self, coords: &[[f32; 2]], ind: [usize; 3], test: usize) -> bool {
        if self.nodes_map[test] == NONE || ind.contains(&test) {
            return false;
        }
        TriIsect::new(ind.map(|i| coords[i])).contains(coords[test])
    }

    /// Blender's `kdtree2d_isect_tri`: the first remaining vertex inside triangle `ind`, or `NONE`.
    fn isect_tri(&self, coords: &[[f32; 2]], ind: [usize; 3]) -> usize {
        let vs = ind.map(|i| coords[i]);
        let bounds = [0, 1].map(|axis| Range {
            min: min3(vs[0][axis], vs[1][axis], vs[2][axis]),
            max: max3(vs[0][axis], vs[1][axis], vs[2][axis]),
        });
        let center = [0, 1].map(|axis| (vs[0][axis] + vs[1][axis] + vs[2][axis]) * (1.0 / 3.0));
        let tri_isect = TriIsect::new(vs);
        self.isect_recursive(coords, ind, &tri_isect, center, bounds, self.root)
    }

    /// Blender's `kdtree2d_isect_tri_recursive`.
    fn isect_recursive(
        &self,
        coords: &[[f32; 2]],
        ind: [usize; 3],
        tri_isect: &TriIsect,
        center: [f32; 2],
        bounds: [Range; 2],
        node_i: usize,
    ) -> usize {
        let node = self.nodes[node_i];
        let co = coords[node.index];
        if !node.removed && tri_isect.contains(co) && !ind.contains(&node.index) {
            return node.index;
        }

        let axis = node.axis;
        let search_pos = || {
            if node.pos != NONE && co[axis] <= bounds[axis].max {
                self.isect_recursive(coords, ind, tri_isect, center, bounds, node.pos)
            } else {
                NONE
            }
        };
        let search_neg = || {
            if node.neg != NONE && co[axis] >= bounds[axis].min {
                self.isect_recursive(coords, ind, tri_isect, center, bounds, node.neg)
            } else {
                NONE
            }
        };

        let pos_first = center[axis] > co[axis];
        let result = if pos_first {
            search_pos()
        } else {
            search_neg()
        };
        if result != NONE {
            result
        } else if pos_first {
            search_neg()
        } else {
            search_pos()
        }
    }
}

/// Blender's `PolyIndex`: a node of the circular list of vertices still in the polygon.
#[derive(Clone, Copy)]
struct PolyIndex {
    next: usize,
    prev: usize,
    index: usize,
    sign: Sign,
    /// Blender's `IndexCacheDirty`: the neighbourhood changed since `last_hit` was cached.
    dirty: bool,
    /// Blender's `index_last_hit`: vertex that last blocked this ear, or `NONE`.
    last_hit: usize,
}

/// Blender's `PolyFill`.
struct PolyFill<'a> {
    coords: &'a [[f32; 2]],
    pis: Vec<PolyIndex>,
    /// Blender's `pf->indices`: any remaining vertex, used as the start of the final triangle.
    head: usize,
    coords_num: usize,
    coords_num_concave: usize,
    tris: Vec<[usize; 3]>,
    /// Built only when the polygon starts out concave (`USE_CONVEX_SKIP`).
    kdtree: Option<KdTree>,
}

impl PolyFill<'_> {
    /// Blender's `pf_coord_sign_calc`.
    fn coord_sign(&self, pi: usize) -> Sign {
        let p = self.pis[pi];
        span_tri_sign(
            self.coords[self.pis[p.prev].index],
            self.coords[p.index],
            self.coords[self.pis[p.next].index],
        )
    }

    /// Recomputes a neighbour of a clipped ear; one that became convex leaves the k-d tree.
    fn refresh_sign(&mut self, pi: usize) {
        let sign = self.coord_sign(pi);
        self.pis[pi].sign = sign;
        if sign == CONVEX {
            self.coords_num_concave -= 1;
            let index = self.pis[pi].index;
            self.kdtree_remove(index);
        }
    }

    fn kdtree_remove(&mut self, index: usize) {
        if let Some(tree) = self.kdtree.as_mut() {
            tree.remove(index);
        }
    }

    /// Blender's `pf_ear_tip_cut` (with `pf_coord_remove`).
    fn ear_tip_cut(&mut self, pi: usize) {
        let ear = self.pis[pi];
        self.tris.push([
            self.pis[ear.prev].index,
            ear.index,
            self.pis[ear.next].index,
        ]);

        self.kdtree_remove(ear.index);
        self.pis[ear.next].prev = ear.prev;
        self.pis[ear.prev].next = ear.next;
        if self.head == pi {
            self.head = ear.next;
        }
        self.coords_num -= 1;
    }

    /// Blender's `pf_ear_tip_check`: whether `pi` is an ear with at least quality `sign_accept`.
    #[expect(
        clippy::expect_used,
        reason = "k-d tree is built whenever concave vertices exist"
    )]
    fn ear_tip_check(&mut self, pi: usize, sign_accept: Sign) -> bool {
        // No concave vertices left: every remaining vertex is a valid ear.
        if self.coords_num_concave == 0 {
            return true;
        }
        if self.pis[pi].sign != sign_accept {
            return false;
        }

        let coords = self.coords;
        let ear = self.pis[pi];
        let ind = [
            ear.index,
            self.pis[ear.next].index,
            self.pis[ear.prev].index,
        ];
        let tree = self
            .kdtree
            .as_ref()
            .expect("k-d tree exists while concave vertices remain");

        if ear.last_hit != NONE {
            if !ear.dirty {
                // Unchanged triangle: the cached blocker still blocks while it is in the tree.
                if tree.nodes_map[ear.last_hit] != NONE {
                    return false;
                }
            } else if tree.isect_single(coords, ind, ear.last_hit) {
                self.pis[pi].dirty = false;
                return false;
            }
        }

        let hit = tree.isect_tri(coords, ind);
        if hit != NONE {
            self.pis[pi].last_hit = hit;
            self.pis[pi].dirty = false;
            return false;
        }
        true
    }

    /// Blender's `pf_ear_tip_find`. Pass one takes convex then tangential ears; the fallback
    /// (degenerate polygons) takes any non-concave vertex, else the start vertex.
    fn ear_tip_find(&mut self, init: usize, reverse: bool) -> usize {
        let n = self.coords_num;
        for sign_accept in [CONVEX, TANGENTIAL] {
            let mut pi = init;
            for _ in 0..n {
                if self.ear_tip_check(pi, sign_accept) {
                    return pi;
                }
                pi = if reverse {
                    self.pis[pi].prev
                } else {
                    self.pis[pi].next
                };
            }
        }

        let mut pi = init;
        for _ in 0..n {
            if self.pis[pi].sign != CONCAVE {
                return pi;
            }
            pi = self.pis[pi].next;
        }
        pi
    }

    /// Blender's `pf_triangulate`.
    fn triangulate(&mut self) {
        let mut pi_ear_init = self.head;
        let mut reverse = false;

        while self.coords_num > 3 {
            let pi_ear = self.ear_tip_find(pi_ear_init, reverse);
            if self.pis[pi_ear].sign != CONVEX {
                self.coords_num_concave -= 1;
            }

            let pi_prev = self.pis[pi_ear].prev;
            let pi_next = self.pis[pi_ear].next;
            self.ear_tip_cut(pi_ear);

            self.pis[pi_prev].dirty = true;
            self.pis[pi_next].dirty = true;

            // The clip removed a vertex; neighbours may have become convex.
            if self.pis[pi_prev].sign != CONVEX {
                self.refresh_sign(pi_prev);
            }
            if self.pis[pi_next].sign != CONVEX {
                self.refresh_sign(pi_next);
            }

            // Sweep: continue from the far side of the new ear, alternating direction past
            // non-convex candidates to avoid lop-sided fans.
            pi_ear_init = if reverse {
                self.pis[pi_prev].prev
            } else {
                self.pis[pi_next].next
            };
            if self.pis[pi_ear_init].sign != CONVEX {
                pi_ear_init = if reverse {
                    self.pis[pi_ear_init].prev
                } else {
                    self.pis[pi_ear_init].next
                };
                reverse = !reverse;
            }
        }

        if self.coords_num == 3 {
            let a = self.head;
            let b = self.pis[a].next;
            let c = self.pis[b].next;
            self.tris
                .push([self.pis[a].index, self.pis[b].index, self.pis[c].index]);
        }
    }
}

/// Blender `BLI_polyfill_calc` (`coords_sign == 1`), with the kd-tree search of `polyfill_2d.cc`:
/// `polyfill_prepare`, then `polyfill_calc`. Returns `n - 2` triangles as indices into `coords`, each
/// keeping the ring winding. Callers pass clockwise projected coordinates.
pub(in crate::geom) fn polyfill(coords: &[[f32; 2]]) -> Vec<[usize; 3]> {
    let n = coords.len();
    let pis = (0..n)
        .map(|i| PolyIndex {
            next: (i + 1) % n,
            prev: (i + n - 1) % n,
            index: i,
            sign: CONVEX,
            dirty: false,
            last_hit: NONE,
        })
        .collect();
    let mut pf = PolyFill {
        coords,
        pis,
        head: 0,
        coords_num: n,
        coords_num_concave: 0,
        tris: Vec::with_capacity(n - 2),
        kdtree: None,
    };

    for i in 0..n {
        let sign = pf.coord_sign(i);
        pf.pis[i].sign = sign;
        if sign != CONVEX {
            pf.coords_num_concave += 1;
        }
    }
    if pf.coords_num_concave != 0 {
        pf.kdtree = Some(KdTree::build(coords, &pf.pis));
    }

    pf.triangulate();
    pf.tris
}
