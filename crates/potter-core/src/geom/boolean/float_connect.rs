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
//!
//! Blender's data structures are reproduced where their order is observable:
//! - Islands come from Blender's group loop, a depth-first walk over each vertex's disk cycle
//!   (`BM_DISK_EDGE_NEXT`), scanning `edge_arr` backwards. The disk order is modelled by [`DiskNet`].
//! - Ray and vertex-visibility casts go through a port of Blender's `BLI_kdopbvh` tree ([`Bvh`]): the
//!   8-ary implicit tree of `BLI_bvhtree_balance`, and the `dfs_raycast` traversal order. When two edges
//!   are at exactly the same distance, the one the traversal reaches first wins, as in Blender.
//!
//! Blender's arm64 FMA contraction is not reproduced; the FLOAT path rounds each operation separately.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::geom::triangulate::ortho_basis;

type P2 = [f32; 2];
type P3 = [f32; 3];
/// Slab bounds of one box, Blender `bv` for `BLI_bvhtree_new(…, axis = 8)`: `[min, max]` for `x`, `y`,
/// `z` and `x + y + z`, in that order.
type Bounds = [f32; 8];

/// Blender `BVH_RAYCAST_DIST_MAX`.
const RAYCAST_DIST_MAX: f32 = f32::MAX / 2.0;
/// Blender `edge_isect_verts_point_2d` endpoint bias.
const ENDPOINT_BIAS: f32 = 1.0e-4;
/// Blender `BLI_bvhtree_new` epsilon (`1e-4f`, above `FLT_EPSILON`), inflating every leaf box.
const BVH_EPSILON: f32 = 1.0e-4;
/// Blender `BLI_bvhtree_new` `tree_type`: children per node of the implicit tree.
const BVH_TREE_TYPE: usize = 8;
/// Blender `bvhtree_kdop_axes[0..4]`. The fourth slab `x + y + z` takes part in overlap tests only; the
/// ray test and the split axis read the first three.
const BVH_AXES: [P3; 4] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
];
/// Blender `bvhtree_kdop_axes_length`: each slab's leaf inflation is `epsilon * length`.
const BVH_AXES_LENGTH: [f32; 4] = [1.0, 1.0, 1.0, 1.732_050_8];
/// Blender `node_minmax_init`.
const EMPTY_BOUNDS: Bounds = [
    f32::MAX,
    -f32::MAX,
    f32::MAX,
    -f32::MAX,
    f32::MAX,
    -f32::MAX,
    f32::MAX,
    -f32::MAX,
];

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

    // The face loop exists before its cut net is built, so loop edges attach to their vertices first.
    let disk = DiskNet::new(n, &edges, (net.len()..edges.len()).chain(0..net.len()));
    let groups = disk.groups();
    if groups.len() < 2 {
        return Vec::new();
    }

    // `axis_dominant_v3_to_m3(f->no)`: rows n1 and n2 of the frame; projection is a dot product.
    let (n1, n2) = ortho_basis(normal);
    // Island spans use raw coordinates, as in Blender's group setup.
    let raw: Vec<P2> = co.iter().map(|&c| [dot3(n1, c), dot3(n2, c)]).collect();
    let mut spans: Vec<Island> = groups
        .into_iter()
        .map(|group| Island::new(group, &raw))
        .collect();
    spans.sort_by(|a, b| {
        axis_pt_cmp(a.min_axis, b.min_axis).then_with(|| axis_pt_cmp(a.max_axis, b.max_axis))
    });
    let mut group = vec![0; n];
    for (rank, island) in spans.iter().enumerate() {
        for &v in &island.members {
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
    let bvh = Bvh::new(
        edges
            .iter()
            .map(|&(a, b)| leaf_bounds(rel[a], rel[b]))
            .collect(),
    );

    let mut state = Islands {
        edges,
        rel,
        group,
        valid: vec![true; n],
        added: Vec::new(),
        bvh,
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

/// The edges of one face's net as Blender's `BMesh` holds them: one entry per `BMEdge`, and the disk
/// cycle (`v->e`) of every vertex.
///
/// Disk order model. `bmesh_disk_edge_append` links a new edge at the tail of the cycle whose head is
/// `v->e` (`bmesh_structure.cc`), and `bm_edge_create` appends to both endpoints (`bmesh_core.cc`). The
/// head therefore stays the first edge attached to the vertex, and `BM_DISK_EDGE_NEXT` visits the edges
/// in attachment order. The model attaches the face loop first (in loop order, since the face's boundary
/// exists before its cut net is built) and then the cut net (in `edge_net_init` order). Connection edges
/// are created after the groups are formed, so they never reach the walk. Two `edge_arr` positions with
/// the same unordered vertex pair are one `BMEdge`, which keeps its first attachment and its stored
/// orientation.
///
/// Evidence: the cycle order is read from `bmesh_structure.cc` and `bmesh_core.cc`. The model's
/// attachment order is not recorded in the Blender records. The walk only changes the result through span
/// ties between distinct vertices of one island that share a projection. In the 24 fixture records no
/// island has such a tie (checked on the face-frame projection), so those records do not exercise it.
struct DiskNet {
    /// Unique edge of each `edge_arr` position.
    of_position: Vec<usize>,
    /// Stored `(v1, v2)` of each unique edge, taken from its first position.
    edges: Vec<(usize, usize)>,
    /// Unique edges of each vertex, in disk cycle order from `v->e`.
    disk: Vec<Vec<usize>>,
}

/// One island of the group loop: its vertices in pop order and its edges in `edge_links` order.
struct Group {
    /// Blender `unique_verts_in_group`, in the order the walk popped them.
    members: Vec<usize>,
    /// Blender `edge_links`: the list head first, which is the most recently discovered edge.
    edge_links: Vec<(usize, usize)>,
}

impl DiskNet {
    /// `creation` lists every `edge_arr` position once, in the order its `BMEdge` was attached.
    fn new(
        vertex_count: usize,
        edges: &[(usize, usize)],
        creation: impl Iterator<Item = usize>,
    ) -> Self {
        let mut of_position = vec![usize::MAX; edges.len()];
        let mut unique: Vec<(usize, usize)> = Vec::new();
        let mut disk: Vec<Vec<usize>> = vec![Vec::new(); vertex_count];
        let mut by_pair: HashMap<(usize, usize), usize> = HashMap::new();
        for position in creation {
            let (a, b) = edges[position];
            let key = (a.min(b), a.max(b));
            let id = if let Some(&id) = by_pair.get(&key) {
                id
            } else {
                let id = unique.len();
                unique.push((a, b));
                disk[a].push(id);
                disk[b].push(id);
                by_pair.insert(key, id);
                id
            };
            of_position[position] = id;
        }
        Self {
            of_position,
            edges: unique,
            disk,
        }
    }

    /// Blender's group loop in `BM_face_split_edgenet_connect_islands`. Each island is a depth-first walk
    /// over the disk cycles from the first unused position, scanning `edge_arr` backwards so the outer
    /// boundary is handled first.
    fn groups(&self) -> Vec<Group> {
        // `EDGE_NOT_IN_STACK` and `VERT_NOT_IN_STACK`: true until the walk reaches the element.
        let mut edge_free = vec![true; self.edges.len()];
        let mut vert_free = vec![true; self.disk.len()];
        let mut groups = Vec::new();
        let mut used = 0;
        let mut position = self.of_position.len() - 1;
        loop {
            let start = self.edges[self.of_position[position]].0;
            vert_free[start] = false;
            let mut stack = vec![start];
            let mut members = Vec::new();
            let mut found = Vec::new();
            while let Some(v) = stack.pop() {
                members.push(v);
                for &e in &self.disk[v] {
                    if !edge_free[e] {
                        continue;
                    }
                    edge_free[e] = false;
                    found.push(e);
                    let (a, b) = self.edges[e];
                    let other = if a == v { b } else { a };
                    if vert_free[other] {
                        vert_free[other] = false;
                        stack.push(other);
                    }
                }
            }
            used += found.len();
            // Blender prepends each edge to `edge_links`, so the list reads newest first.
            let edge_links: Vec<(usize, usize)> =
                found.iter().rev().map(|&e| self.edges[e]).collect();
            groups.push(Group {
                members,
                edge_links,
            });
            if used == self.edges.len() {
                return groups;
            }
            // Skip positions whose edge already belongs to a group.
            while !edge_free[self.of_position[position]] {
                position -= 1;
            }
        }
    }
}

/// Blender `EdgeGroupIsland` before sorting: the member vertices and their projection span.
struct Island {
    /// Member vertices, in the walk's pop order.
    members: Vec<usize>,
    /// Lowest vertex by raw projection (`vert_span.min`).
    min: usize,
    /// Highest vertex by raw projection (`vert_span.max`).
    max: usize,
    min_axis: P2,
    max_axis: P2,
}

impl Island {
    /// Blender's span setup in `BM_face_split_edgenet_connect_islands`. The scan follows `edge_links`
    /// order, and the first vertex to reach each extreme keeps it on ties.
    fn new(group: Group, raw: &[P2]) -> Self {
        let Group {
            members,
            edge_links,
        } = group;
        let (head_min, head_max) = edge_links[0];
        let mut min = head_min;
        let mut max = head_max;
        let mut min_axis = [f32::MAX, f32::MAX];
        let mut max_axis = [-f32::MAX, -f32::MAX];
        for &(a, b) in &edge_links {
            for v in [a, b] {
                if axis_pt_cmp(raw[v], min_axis) == Ordering::Less {
                    min = v;
                    min_axis = raw[v];
                }
                if axis_pt_cmp(raw[v], max_axis) == Ordering::Greater {
                    max = v;
                    max_axis = raw[v];
                }
            }
        }
        Self {
            members,
            min,
            max,
            min_axis,
            max_axis,
        }
    }
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
    /// Blender `edge_arr` BVH: leaf `i` is `edges[i]`.
    bvh: Bvh,
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
        let mut hit = RayHit {
            index: None,
            dist: RAYCAST_DIST_MAX,
        };
        self.bvh.ray_cast(
            [o[0], o[1], 0.0],
            [dir[0], dir[1], 0.0],
            &mut hit,
            &mut |id: usize, best: &mut RayHit| {
                let e = self.edges[id];
                if let Some(dist) = isect_ray_seg(o, dir, self.rel[e.0], self.rel[e.1])
                    && dist < best.dist
                    && dist > 0.0
                    && e.0 != origin
                    && e.1 != origin
                    && self.group[e.0] != island
                {
                    best.dist = dist;
                    best.index = Some(id);
                }
            },
        );
        // Blender checks the new edges only when the BVH found a hit.
        let id = hit.index?;
        let mut best = hit.dist;
        let mut e_hit = self.edges[id];
        for &e in &self.added {
            if let Some(dist) = isect_ray_seg(o, dir, self.rel[e.0], self.rel[e.1])
                && e.0 != origin
                && e.1 != origin
                && dist < best
            {
                best = dist;
                e_hit = e;
            }
        }
        Some(e_hit)
    }

    /// Blender `test_edges_isect_2d_vert`: the edge that blocks the segment `origin -> other`.
    fn vert_hit(&self, origin: usize, other: usize, island: usize) -> Option<(usize, usize)> {
        let a = self.rel[origin];
        let b = self.rel[other];
        let mut dir = sub2(b, a);
        let dist_orig = normalize2(&mut dir);
        let mut hit = RayHit {
            index: None,
            dist: dist_orig,
        };
        self.bvh.ray_cast(
            [a[0], a[1], 0.0],
            [dir[0], dir[1], 0.0],
            &mut hit,
            &mut |id: usize, best: &mut RayHit| {
                let e = self.edges[id];
                if let Some(co) = self.vert_edge_isect(e, origin, other) {
                    let dist_new = dist_orig * line_point_factor(co, a, b);
                    if dist_new < best.dist && dist_new > 0.0 && self.group[e.0] != island {
                        best.dist = dist_new;
                        best.index = Some(id);
                    }
                }
            },
        );
        if let Some(id) = hit.index {
            return Some(self.edges[id]);
        }
        // Blender checks the new edges only when the BVH found no blocker.
        let mut t_best = 1.0;
        let mut found = None;
        for &e in &self.added {
            if let Some(co) = self.vert_edge_isect(e, origin, other) {
                let t = line_point_factor(co, a, b);
                if t < t_best {
                    t_best = t;
                    found = Some(e);
                }
            }
        }
        found
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

/// Blender `BVHRayCastData::hit`: `index` stays `None` (Blender `-1`) until a callback accepts an edge.
struct RayHit {
    index: Option<usize>,
    dist: f32,
}

/// Blender `BVHNode`'s child link: a leaf (edge index) or a branch (branch index).
#[derive(Clone, Copy)]
enum Child {
    Leaf(usize),
    Branch(usize),
}

/// Blender `BVHNode` for a branch: its box, its split axis (a slab index), and its `node_num` children.
#[derive(Clone, Default)]
struct BvhBranch {
    bounds: Bounds,
    /// Blender `main_axis`: the slab (0, 1, 2) of the split axis.
    main_axis: usize,
    children: Vec<Child>,
}

/// Blender `BVHTree` over the edge list: `BLI_bvhtree_insert` for each edge in `edge_arr` order, then
/// `BLI_bvhtree_balance`. Leaf `i` is edge `i`.
struct Bvh {
    leaves: Vec<Bounds>,
    /// Branch `j` is stored at index `j`, Blender's implicit numbering. The root is branch 1.
    branches: Vec<BvhBranch>,
}

/// Blender `BVHBuildHelper`: the layout of `leafs_num` leaves in an implicit `BVH_TREE_TYPE`-ary tree.
struct Implicit {
    leafs_num: usize,
    leafs_per_child: [usize; 32],
    branches_on_level: [usize; 32],
    remain_leafs: usize,
}

impl Implicit {
    /// Blender `build_implicit_tree_helper`.
    fn new(leafs_num: usize) -> Self {
        let mut leafs_per_child = [0usize; 32];
        let mut branches_on_level = [0usize; 32];
        leafs_per_child[0] = 1;
        while leafs_per_child[0] < leafs_num {
            leafs_per_child[0] *= BVH_TREE_TYPE;
        }
        branches_on_level[0] = 1;
        let mut depth = 1;
        while depth < 32 && leafs_per_child[depth - 1] != 0 {
            branches_on_level[depth] = branches_on_level[depth - 1] * BVH_TREE_TYPE;
            leafs_per_child[depth] = leafs_per_child[depth - 1] / BVH_TREE_TYPE;
            depth += 1;
        }
        let remain = leafs_num - leafs_per_child[1];
        let nnodes = (remain + BVH_TREE_TYPE - 2) / (BVH_TREE_TYPE - 1);
        Self {
            leafs_num,
            leafs_per_child,
            branches_on_level,
            remain_leafs: remain + nnodes,
        }
    }

    /// Blender `implicit_leafs_index`: the first leaf-array position under the branch `child_index` of
    /// the level above `depth`.
    fn leafs_index(&self, depth: usize, child_index: usize) -> usize {
        let min_leaf_index = child_index * self.leafs_per_child[depth - 1];
        if min_leaf_index <= self.remain_leafs {
            return min_leaf_index;
        }
        if self.leafs_per_child[depth] != 0 {
            return self.leafs_num
                - (self.branches_on_level[depth - 1] - child_index) * self.leafs_per_child[depth];
        }
        self.remain_leafs
    }
}

/// One level of the implicit tree while it is divided.
#[derive(Clone, Copy)]
struct Level {
    /// Index of the first branch on this level.
    start: usize,
    /// Index of the first branch on the next level (Blender `first_of_next_level`).
    next: usize,
    /// Blender `depth`.
    depth: usize,
}

/// Blender `i * tree_type + tree_offset + k`, with `tree_offset = 2 - tree_type`.
fn child_index(parent: usize, k: usize) -> usize {
    parent * BVH_TREE_TYPE + k + 2 - BVH_TREE_TYPE
}

/// Blender `create_kdop_hull` and `bvhtree_node_inflate` for one edge's two points (`z = 0`).
fn leaf_bounds(a: P2, b: P2) -> Bounds {
    kdop_bounds(&[[a[0], a[1], 0.0], [b[0], b[1], 0.0]], BVH_EPSILON)
}

/// Blender `BLI_bvhtree_insert` leaf box of `points` (`create_kdop_hull`, then `bvhtree_node_inflate`
/// with `epsilon`).
pub(super) fn kdop_bounds(points: &[P3], epsilon: f32) -> Bounds {
    let mut bounds = EMPTY_BOUNDS;
    for &p in points {
        extend_bounds(&mut bounds, p);
    }
    for (axis, &length) in BVH_AXES_LENGTH.iter().enumerate() {
        let dist = epsilon * length;
        bounds[2 * axis] -= dist;
        bounds[2 * axis + 1] += dist;
    }
    bounds
}

/// Blender `create_kdop_hull` for one point.
fn extend_bounds(bounds: &mut Bounds, p: P3) {
    for (axis, &axis_dir) in BVH_AXES.iter().enumerate() {
        let d = dot3(p, axis_dir);
        bounds[2 * axis] = d.min(bounds[2 * axis]);
        bounds[2 * axis + 1] = d.max(bounds[2 * axis + 1]);
    }
}

/// Blender `refit_kdop_hull`: the union of the boxes of `ids`.
fn refit(leaves: &[Bounds], ids: &[usize]) -> Bounds {
    let mut bounds = EMPTY_BOUNDS;
    for &id in ids {
        for axis in 0..4 {
            bounds[2 * axis] = leaves[id][2 * axis].min(bounds[2 * axis]);
            bounds[2 * axis + 1] = leaves[id][2 * axis + 1].max(bounds[2 * axis + 1]);
        }
    }
    bounds
}

/// Blender `get_largest_axis`: the largest-extent axis, as its max slab index (1, 3 or 5).
fn largest_axis(bv: &Bounds) -> usize {
    let m0 = bv[1] - bv[0];
    let m1 = bv[3] - bv[2];
    let m2 = bv[5] - bv[4];
    if m0 > m1 {
        if m0 > m2 { 1 } else { 5 }
    } else if m1 > m2 {
        3
    } else {
        5
    }
}

/// Blender `bvh_medianof3`: the median of three leaves by their max slab on `axis`.
fn median_of3(
    a: &[usize],
    lo: usize,
    mid: usize,
    hi: usize,
    axis: usize,
    leaves: &[Bounds],
) -> usize {
    let key = |id: usize| leaves[id][axis];
    if key(a[mid]) < key(a[lo]) {
        if key(a[hi]) < key(a[mid]) {
            return a[mid];
        }
        if key(a[hi]) < key(a[lo]) {
            return a[hi];
        }
        return a[lo];
    }
    if key(a[hi]) < key(a[mid]) {
        if key(a[hi]) < key(a[lo]) {
            return a[lo];
        }
        return a[hi];
    }
    a[mid]
}

/// Blender `bvh_partition`: Hoare partition of `a[lo..hi]` around the pivot key.
fn partition(
    a: &mut [usize],
    lo: usize,
    hi: usize,
    pivot_key: f32,
    axis: usize,
    leaves: &[Bounds],
) -> usize {
    let key = |id: usize| leaves[id][axis];
    let mut i = lo;
    let mut j = hi;
    loop {
        while key(a[i]) < pivot_key {
            i += 1;
        }
        j -= 1;
        while pivot_key < key(a[j]) {
            j -= 1;
        }
        if i >= j {
            return i;
        }
        a.swap(i, j);
        i += 1;
    }
}

/// Blender `bvh_insertionsort`.
fn insertion_sort(a: &mut [usize], lo: usize, hi: usize, axis: usize, leaves: &[Bounds]) {
    for i in lo..hi {
        let mut j = i;
        let t = a[i];
        while j != lo && leaves[t][axis] < leaves[a[j - 1]][axis] {
            a[j] = a[j - 1];
            j -= 1;
        }
        a[j] = t;
    }
}

/// Blender `partition_nth_element`: after it, every leaf left of `n` is at most the leaf at `n`.
fn partition_nth_element(
    a: &mut [usize],
    mut begin: usize,
    mut end: usize,
    n: usize,
    axis: usize,
    leaves: &[Bounds],
) {
    while end - begin > 3 {
        let pivot = median_of3(a, begin, usize::midpoint(begin, end), end - 1, axis, leaves);
        let cut = partition(a, begin, end, leaves[pivot][axis], axis, leaves);
        if cut <= n {
            begin = cut;
        } else {
            end = cut;
        }
    }
    insertion_sort(a, begin, end, axis, leaves);
}

/// Blender `split_leafs`: each child range gets the leaves it would get from a full sort on `axis`.
fn split_leafs(order: &mut [usize], nth: &[usize], axis: usize, leaves: &[Bounds]) {
    for i in 0..BVH_TREE_TYPE - 1 {
        if nth[i] >= nth[BVH_TREE_TYPE] {
            break;
        }
        partition_nth_element(order, nth[i], nth[BVH_TREE_TYPE], nth[i + 1], axis, leaves);
    }
}

/// Blender `non_recursive_bvh_div_nodes_task_cb`: fits branch `j`, splits its leaves along its largest
/// axis, and links its children.
fn divide(
    branches: &mut [BvhBranch],
    order: &mut [usize],
    leaves: &[Bounds],
    implicit: &Implicit,
    level: Level,
    j: usize,
) {
    let parent_level_index = j - level.start;
    let begin = implicit.leafs_index(level.depth, parent_level_index);
    let end = implicit.leafs_index(level.depth, parent_level_index + 1);
    let bounds = refit(leaves, &order[begin..end]);
    let split_axis = largest_axis(&bounds);

    let mut nth = [0usize; BVH_TREE_TYPE + 1];
    nth[0] = begin;
    nth[BVH_TREE_TYPE] = end;
    for (k, slot) in nth.iter_mut().enumerate().take(BVH_TREE_TYPE).skip(1) {
        let child_level_index = child_index(j, k) - level.next;
        *slot = implicit.leafs_index(level.depth + 1, child_level_index);
    }
    split_leafs(order, &nth, split_axis, leaves);

    let mut children = Vec::with_capacity(BVH_TREE_TYPE);
    for k in 0..BVH_TREE_TYPE {
        let index = child_index(j, k);
        let child_level_index = index - level.next;
        let first = implicit.leafs_index(level.depth + 1, child_level_index);
        let last = implicit.leafs_index(level.depth + 1, child_level_index + 1);
        match last.cmp(&(first + 1)) {
            Ordering::Greater => children.push(Child::Branch(index)),
            Ordering::Equal => children.push(Child::Leaf(order[first])),
            Ordering::Less => break,
        }
    }
    branches[j] = BvhBranch {
        bounds,
        main_axis: split_axis / 2,
        children,
    };
}

/// Blender `RayPrecalc` fields of `bvhtree_ray_cast_data_precalc`, for the three slabs.
struct RayPrecalc {
    /// Blender `ray_dot_axis`: the ray's dot product with each slab direction, zeroed when tiny.
    dot: [f32; 3],
    /// Blender `idot_axis`.
    idot: [f32; 3],
    /// Blender `index`: the bounds index of the near and far plane of each slab.
    index: [usize; 6],
}

impl RayPrecalc {
    fn new(dir: P3) -> Self {
        let mut pre = Self {
            dot: [0.0; 3],
            idot: [0.0; 3],
            index: [0; 6],
        };
        for (axis, &axis_dir) in BVH_AXES.iter().take(3).enumerate() {
            let d = dot3(dir, axis_dir);
            if d.abs() < f32::EPSILON {
                pre.dot[axis] = 0.0;
                // The sign does not matter here, because `index` is adjusted below anyway.
                pre.idot[axis] = f32::MAX;
            } else {
                pre.dot[axis] = d;
                pre.idot[axis] = 1.0 / d;
            }
            let near = usize::from(pre.idot[axis] < 0.0);
            pre.index[2 * axis] = near + 2 * axis;
            pre.index[2 * axis + 1] = (1 - near) + 2 * axis;
        }
        pre
    }
}

/// Blender `fast_ray_nearest_hit` with zero radius: the slab entry distance of `bv`, or `f32::MAX` on a
/// miss or when the box lies beyond `hit_dist`.
fn fast_ray_nearest_hit(bv: &Bounds, origin: P3, pre: &RayPrecalc, hit_dist: f32) -> f32 {
    let t1x = (bv[pre.index[0]] - origin[0]) * pre.idot[0];
    let t2x = (bv[pre.index[1]] - origin[0]) * pre.idot[0];
    let t1y = (bv[pre.index[2]] - origin[1]) * pre.idot[1];
    let t2y = (bv[pre.index[3]] - origin[1]) * pre.idot[1];
    let t1z = (bv[pre.index[4]] - origin[2]) * pre.idot[2];
    let t2z = (bv[pre.index[5]] - origin[2]) * pre.idot[2];

    if (t1x > t2y || t2x < t1y || t1x > t2z || t2x < t1z || t1y > t2z || t2y < t1z)
        || (t2x < 0.0 || t2y < 0.0 || t2z < 0.0)
        || (t1x > hit_dist || t1y > hit_dist || t1z > hit_dist)
    {
        return f32::MAX;
    }
    t1x.max(t1y).max(t1z)
}

impl Bvh {
    /// Blender `BLI_bvhtree_balance` (`non_recursive_bvh_div_nodes`) over the leaf boxes, in insertion order.
    fn new(leaves: Vec<Bounds>) -> Self {
        let leafs_num = leaves.len();
        let branches_num = ((leafs_num + BVH_TREE_TYPE - 3) / (BVH_TREE_TYPE - 1)).max(1);
        let mut branches = vec![BvhBranch::default(); branches_num + 1];
        if leafs_num == 1 {
            branches[1] = BvhBranch {
                bounds: leaves[0],
                main_axis: largest_axis(&leaves[0]) / 2,
                children: vec![Child::Leaf(0)],
            };
            return Self { leaves, branches };
        }

        let implicit = Implicit::new(leafs_num);
        let mut order: Vec<usize> = (0..leafs_num).collect();
        let mut start = 1;
        let mut depth = 1;
        while start <= branches_num {
            let next = child_index(start, 0);
            let stop = next.min(branches_num + 1);
            let level = Level { start, next, depth };
            for j in start..stop {
                divide(&mut branches, &mut order, &leaves, &implicit, level, j);
            }
            start = next;
            depth += 1;
        }
        Self { leaves, branches }
    }

    /// Blender `BLI_bvhtree_ray_cast_ex` with zero radius: `dfs_raycast` from the root. `callback` is
    /// called for each leaf whose box can still hold a closer hit, and may update `hit`.
    fn ray_cast(
        &self,
        origin: P3,
        dir: P3,
        hit: &mut RayHit,
        callback: &mut dyn FnMut(usize, &mut RayHit),
    ) {
        let pre = RayPrecalc::new(dir);
        self.dfs(Child::Branch(1), origin, &pre, hit, callback);
    }

    fn dfs(
        &self,
        child: Child,
        origin: P3,
        pre: &RayPrecalc,
        hit: &mut RayHit,
        callback: &mut dyn FnMut(usize, &mut RayHit),
    ) {
        match child {
            Child::Leaf(id) => {
                if fast_ray_nearest_hit(&self.leaves[id], origin, pre, hit.dist) >= hit.dist {
                    return;
                }
                callback(id, hit);
            }
            Child::Branch(j) => {
                let node = &self.branches[j];
                if fast_ray_nearest_hit(&node.bounds, origin, pre, hit.dist) >= hit.dist {
                    return;
                }
                // Dive in the direction of the ray along the split axis.
                if pre.dot[node.main_axis] > 0.0 {
                    for &next in &node.children {
                        self.dfs(next, origin, pre, hit, callback);
                    }
                } else {
                    for &next in node.children.iter().rev() {
                        self.dfs(next, origin, pre, hit, callback);
                    }
                }
            }
        }
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

/// Blender `BLI_bvhtree_overlap_ex` over two `BLI_bvhtree_new(…, 8, 8)` trees: both balanced as
/// [`Bvh::new`], then `tree_overlap_traverse` from the roots. Each pair is `(leaf of first, leaf of second)`
/// as positions in the given box lists. Blender's `BLI_stack_pop_n` hands the pushed pairs back last-first.
pub(super) fn overlap_pairs(first: Vec<Bounds>, second: Vec<Bounds>) -> Vec<(usize, usize)> {
    if first.is_empty() || second.is_empty() {
        return Vec::new();
    }
    let tree_a = Bvh::new(first);
    let tree_b = Bvh::new(second);
    let mut out = Vec::new();
    overlap_traverse(
        &tree_a,
        Child::Branch(1),
        &tree_b,
        Child::Branch(1),
        &mut out,
    );
    out.reverse();
    out
}

/// Bounds of a child node: its leaf box, or its branch box.
fn child_bounds(tree: &Bvh, child: Child) -> Bounds {
    match child {
        Child::Leaf(id) => tree.leaves[id],
        Child::Branch(index) => tree.branches[index].bounds,
    }
}

/// Blender `tree_overlap_traverse`: while the boxes overlap, descend the first node while it is a branch,
/// otherwise the second node's children, and emit a pair when both are leaves.
fn overlap_traverse(
    tree1: &Bvh,
    node1: Child,
    tree2: &Bvh,
    node2: Child,
    out: &mut Vec<(usize, usize)>,
) {
    let a = child_bounds(tree1, node1);
    let b = child_bounds(tree2, node2);
    if !(0..4).all(|axis| a[2 * axis] <= b[2 * axis + 1] && b[2 * axis] <= a[2 * axis + 1]) {
        return;
    }
    match (node1, node2) {
        (Child::Leaf(id1), Child::Leaf(id2)) => out.push((id1, id2)),
        (Child::Leaf(_), Child::Branch(index)) => {
            for &child in &tree2.branches[index].children {
                overlap_traverse(tree1, node1, tree2, child, out);
            }
        }
        (Child::Branch(index), _) => {
            for &child in &tree1.branches[index].children {
                overlap_traverse(tree1, child, tree2, node2, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Bvh, Child, DiskNet, Island, P2, RAYCAST_DIST_MAX, RayHit, connect_islands, isect_ray_seg,
        leaf_bounds,
    };
    use serde_json::Value;

    const FIXTURE: &str = include_str!(
        "../../../../../tests/fixtures/boolean_parity/connect_islands/coplanar_float.json"
    );

    const TIE_FIXTURE: &str = include_str!(
        "../../../../../tests/fixtures/boolean_parity/connect_islands/bvh_tie_float.json"
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

    /// Checks every `ok` record of a connect fixture: `connect_islands` must return the recorded connect
    /// pairs, compared unordered (Blender returns pre-existing edges in their stored orientation).
    /// Returns the number of records checked and how many of them connected.
    fn check_connect_fixture(fixture: &str) -> (usize, usize) {
        let Ok(root) = serde_json::from_str::<Value>(fixture) else {
            panic!("connect_islands fixture is valid JSON");
        };
        let Some(cases) = root["cases"].as_object() else {
            panic!("connect_islands fixture has a cases object");
        };
        let mut checked = 0;
        let mut connected = 0;
        for (name, records) in cases {
            let Some(records) = records.as_array() else {
                panic!("{name}: records must be an array");
            };
            for (number, record) in records.iter().enumerate() {
                if record["ok"].as_bool() != Some(true) {
                    continue;
                }
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
                checked += 1;
                connected += usize::from(!expected.is_empty());
            }
        }
        (checked, connected)
    }

    #[test]
    fn connect_islands_matches_blender_records_exactly() {
        let (checked, connected) = check_connect_fixture(FIXTURE);
        assert_eq!(checked, 24, "fixture holds 24 ok records");
        assert!(connected > 0, "fixture exercises at least one connection");
    }

    /// The Blender record in `bvh_tie_float.json`: the top face of a union, where the connect origin's
    /// +x ray first meets a notch tip shared by two edges at exactly the same distance (0.75). Blender
    /// keeps the edge its BVH traversal reaches first, and the port must do the same.
    #[test]
    fn connect_bvh_tie_blender_record_matches() {
        let (checked, connected) = check_connect_fixture(TIE_FIXTURE);
        assert_eq!(checked, 1, "tie fixture holds one ok record");
        assert!(connected > 0, "tie fixture exercises a connection");
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

    /// Edge positions 0..: net `(1,3)`, loop `(0,1), (1,2), (2,0)`. Creation order is loop then net.
    /// Unique ids follow creation: `(0,1)=0`, `(1,2)=1`, `(2,0)=2`, `(1,3)=3`.
    fn triangle_with_spur() -> DiskNet {
        let edges = [(1, 3), (0, 1), (1, 2), (2, 0)];
        DiskNet::new(4, &edges, (1..4).chain(0..1))
    }

    #[test]
    fn connect_disk_order_creation_is_loop_then_net() {
        // Net (4,5) is position 0; the loop square 0-1-2-3-0 is positions 1..=4.
        let edges = [(4, 5), (0, 1), (1, 2), (2, 3), (3, 0)];
        let disk = DiskNet::new(6, &edges, (1..5).chain(0..1));
        assert_eq!(disk.of_position, vec![4, 0, 1, 2, 3]);
        // Vertex 0 is attached to (0,1) first, then (3,0). The net edge comes last on vertex 4.
        assert_eq!(disk.disk[0], vec![0, 3]);
        assert_eq!(disk.disk[1], vec![0, 1]);
        assert_eq!(disk.disk[4], vec![4]);
    }

    #[test]
    fn connect_disk_order_walk_matches_blender_dfs() {
        let groups = triangle_with_spur().groups();
        assert_eq!(groups.len(), 1);
        // Start at position 3, edge (2,0). Walk: pop 2 (edges (1,2), (2,0)), then pop 0 (edge (0,1)),
        // then pop 1 (edge (1,3)), then pop 3.
        assert_eq!(groups[0].members, vec![2, 0, 1, 3]);
        // Discovery order is (1,2), (2,0), (0,1), (1,3); `edge_links` is newest first.
        assert_eq!(groups[0].edge_links, vec![(1, 3), (0, 1), (2, 0), (1, 2)]);
    }

    #[test]
    fn connect_disk_order_walk_scans_edge_arr_backwards() {
        // Net (3,4) is position 0, the loop triangle is positions 1..=3, so the first walk starts on
        // the triangle and the second on the spur at position 0.
        let edges = [(3, 4), (0, 1), (1, 2), (2, 0)];
        let groups = DiskNet::new(5, &edges, (1..4).chain(0..1)).groups();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].members, vec![2, 0, 1]);
        assert_eq!(groups[0].edge_links, vec![(0, 1), (2, 0), (1, 2)]);
        assert_eq!(groups[1].members, vec![3, 4]);
        assert_eq!(groups[1].edge_links, vec![(3, 4)]);
    }

    #[test]
    fn connect_disk_order_span_tie_takes_first_visit() {
        // Vertices 0 and 3 share a projection. The walk visits 3 before 0 (edge (1,3) heads
        // `edge_links`), so Blender's strict span scan keeps 3 as the minimum.
        let group = triangle_with_spur().groups().remove(0);
        let raw: [P2; 4] = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 0.0]];
        let island = Island::new(group, &raw);
        assert_eq!((island.min, island.max), (3, 2));
    }

    /// Nearest hit along a ray over `segments`, with the connect predicates (`0 < dist < best`).
    fn cast(segments: &[(P2, P2)], origin: P2, dir: P2) -> Option<usize> {
        let bvh = Bvh::new(segments.iter().map(|&(a, b)| leaf_bounds(a, b)).collect());
        let mut hit = RayHit {
            index: None,
            dist: RAYCAST_DIST_MAX,
        };
        bvh.ray_cast(
            [origin[0], origin[1], 0.0],
            [dir[0], dir[1], 0.0],
            &mut hit,
            &mut |id: usize, best: &mut RayHit| {
                let (a, b) = segments[id];
                if let Some(dist) = isect_ray_seg(origin, dir, a, b)
                    && dist < best.dist
                    && dist > 0.0
                {
                    best.dist = dist;
                    best.index = Some(id);
                }
            },
        );
        hit.index
    }

    /// Both segments cross the ray at exactly t = 1. The root is split on x (its largest extent), and
    /// both leaves have the same max-x key, so the stable sort keeps leaf order `[0, 1]`. A ray with
    /// positive x direction dives into the children in that order, so leaf 0 is reached first and keeps
    /// the tie.
    #[test]
    fn connect_bvh_tie_forward_first_child_wins() {
        let segments = [([0.25, -0.5], [1.75, 0.5]), ([0.25, 0.5], [1.75, -0.5])];
        assert_eq!(cast(&segments, [0.0, 0.0], [1.0, 0.0]), Some(0));
    }

    /// The same two segments, cast along -x from the other side. The ray's dot with the split axis is
    /// negative, so the children are visited in reverse order and leaf 1 is reached first.
    #[test]
    fn connect_bvh_tie_reverse_last_child_wins() {
        let segments = [([0.25, -0.5], [1.75, 0.5]), ([0.25, 0.5], [1.75, -0.5])];
        assert_eq!(cast(&segments, [2.0, 0.0], [-1.0, 0.0]), Some(1));
    }

    fn leaves_under(bvh: &Bvh, child: Child, out: &mut Vec<usize>) {
        match child {
            Child::Leaf(id) => out.push(id),
            Child::Branch(j) => {
                for &next in &bvh.branches[j].children {
                    leaves_under(bvh, next, out);
                }
            }
        }
    }

    /// The implicit 8-ary tree reaches every leaf exactly once, and the nearest hit matches a brute-force
    /// scan on tie-free random rays.
    #[test]
    fn connect_bvh_tree_matches_brute_force() {
        let mut seed = 0x2545_f491_u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / 16_777_216.0 * 10.0 - 5.0
        };
        let segments: Vec<(P2, P2)> = (0..40)
            .map(|_| ([next(), next()], [next(), next()]))
            .collect();

        let bvh = Bvh::new(segments.iter().map(|&(a, b)| leaf_bounds(a, b)).collect());
        let mut reached = Vec::new();
        leaves_under(&bvh, Child::Branch(1), &mut reached);
        reached.sort_unstable();
        assert_eq!(reached, (0..segments.len()).collect::<Vec<_>>());

        for _ in 0..200 {
            let origin = [next(), next()];
            let raw = [next(), next()];
            let len = (raw[0] * raw[0] + raw[1] * raw[1]).sqrt();
            let dir = [raw[0] / len, raw[1] / len];
            let mut expected: Option<(usize, f32)> = None;
            for (id, &(a, b)) in segments.iter().enumerate() {
                if let Some(dist) = isect_ray_seg(origin, dir, a, b)
                    && dist > 0.0
                    && expected.is_none_or(|(_, best)| dist < best)
                {
                    expected = Some((id, dist));
                }
            }
            assert_eq!(cast(&segments, origin, dir), expected.map(|(id, _)| id));
        }
    }
}
