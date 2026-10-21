//! Blender's k-DOP bounding-volume tree over triangles, ported in `f32` from `BLI_kdopbvh.cc`
//! (pinned 5.2.2), with the watertight ray cast of `BLI_bvhtree_ray_cast_ex`.
//!
//! The tree is the one `bvhtree_new_common` (`bvhutils.cc`) builds with
//! `BLI_bvhtree_new(n, 0.0, 2, 6)`: a binary tree with three AABB slabs, and its epsilon is
//! clamped to `FLT_EPSILON` (`BLI_kdopbvh.cc`). Leaves are inserted in order and built by
//! `BLI_bvhtree_balance` (`non_recursive_bvh_div_nodes`, `split_leafs`, `partition_nth_element`,
//! `get_largest_axis`). The ray cast is `dfs_raycast` with `BVH_RAYCAST_WATERTIGHT`
//! (`fast_ray_nearest_hit`). The nearest-point search is `dfs_find_nearest_dfs` and, with
//! `BVH_NEAREST_OPTIMAL_ORDER`, `heap_find_nearest_inner` over `BLI_heapsimple`. The tree knows
//! nothing about its leaves: the caller tests each leaf index it is called back with.

use crate::geom::heap_simple::HeapSimple;
/// A point or direction in `f32`, as Blender stores them.
pub(crate) type Vec3 = [f32; 3];

/// Unit axes `bvhtree_kdop_axes[0..3]`.
const AXES: [Vec3; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// `BLI_bvhtree_new` clamps the tree epsilon to `FLT_EPSILON`; leaf boxes are inflated by it.
const TREE_EPSILON: f32 = f32::EPSILON;
pub(crate) fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `axis_dominant_v3_single` (`BLI_math_geom_inline.cc`): the axis of largest magnitude.
fn axis_dominant(v: Vec3) -> usize {
    let (x, y, z) = (v[0].abs(), v[1].abs(), v[2].abs());
    if x > y {
        if x > z { 0 } else { 2 }
    } else if y > z {
        1
    } else {
        2
    }
}

/// Leaf box of one triangle: its extent on x, y and z (`create_kdop_hull`), inflated by the tree epsilon
/// (`bvhtree_node_inflate`). The axis lengths are 1 for x, y and z.
fn triangle_bounds(tri: [Vec3; 3]) -> [f32; 6] {
    let mut bounds = [0.0_f32; 6];
    for axis in 0..3 {
        let mut min = f32::MAX;
        let mut max = -f32::MAX;
        for point in tri {
            min = min.min(point[axis]);
            max = max.max(point[axis]);
        }
        bounds[2 * axis] = min - TREE_EPSILON;
        bounds[2 * axis + 1] = max + TREE_EPSILON;
    }
    bounds
}

/// Blender's `BuildHelper` (`BLI_kdopbvh.cc`, `build_implicit_tree_helper`) for a binary tree.
struct BuildHelper {
    leafs_num: i64,
    leafs_per_child: [i64; 32],
    branches_on_level: [i64; 32],
    remain_leafs: i64,
}

impl BuildHelper {
    fn new(leafs_num: usize) -> Self {
        let n = leafs_num as i64;
        let mut helper = Self {
            leafs_num: n,
            leafs_per_child: [0; 32],
            branches_on_level: [0; 32],
            remain_leafs: 0,
        };
        let mut per_child = 1_i64;
        while per_child < n {
            per_child *= 2;
        }
        helper.leafs_per_child[0] = per_child;
        helper.branches_on_level[0] = 1;
        let mut depth = 1;
        while depth < 32 && helper.leafs_per_child[depth - 1] != 0 {
            helper.branches_on_level[depth] = helper.branches_on_level[depth - 1] * 2;
            helper.leafs_per_child[depth] = helper.leafs_per_child[depth - 1] / 2;
            depth += 1;
        }
        let remain = n - helper.leafs_per_child[1];
        // Blender: nnodes = (remain + K - 2) / (K - 1), which is `remain` for K = 2.
        let nnodes = remain;
        helper.remain_leafs = remain + nnodes;
        helper
    }

    /// `implicit_leafs_index`: the first leaf position reachable from branch `child_index` on `depth`.
    fn leafs_index(&self, depth: i64, child_index: i64) -> i64 {
        let depth = depth as usize;
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

/// A node of the cage tree: a leaf (triangle index) or a branch (index into the branch array).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NodeRef {
    Leaf(usize),
    Branch(usize),
}

/// A branch node: its box over every leaf below it, up to two children, and its split axis.
#[derive(Clone, Copy)]
struct Branch {
    bv: [f32; 6],
    children: [Option<NodeRef>; 2],
    node_num: usize,
    main_axis: usize,
}

impl Branch {
    const EMPTY: Self = Self {
        bv: [0.0; 6],
        children: [None, None],
        node_num: 0,
        main_axis: 0,
    };
}

/// `get_largest_axis` (`BLI_kdopbvh.cc`): the bound index of the largest extent (1, 3 or 5, i.e. the
/// axis's maximum slot). Ties go to the later axis, as in Blender.
fn largest_axis(bv: &[f32; 6]) -> usize {
    let (m0, m1, m2) = (bv[1] - bv[0], bv[3] - bv[2], bv[5] - bv[4]);
    if m0 > m1 {
        if m0 > m2 { 1 } else { 5 }
    } else if m1 > m2 {
        3
    } else {
        5
    }
}

/// The balance state of `non_recursive_bvh_div_nodes`: the leaf order is permuted in place as branches split.
struct TreeBuilder<'a> {
    leaf_bv: &'a [[f32; 6]],
    leafs: Vec<usize>,
    helper: BuildHelper,
    branches: Vec<Branch>,
}

impl TreeBuilder<'_> {
    fn key(&self, position: i64, axis: usize) -> f32 {
        self.leaf_bv[self.leafs[position as usize]][axis]
    }

    /// `bvh_medianof3`: the median of three keys.
    fn median_of_3(&self, lo: i64, mid: i64, hi: i64, axis: usize) -> f32 {
        let (lo_key, mid_key, hi_key) =
            (self.key(lo, axis), self.key(mid, axis), self.key(hi, axis));
        if mid_key < lo_key {
            if hi_key < mid_key {
                mid_key
            } else if hi_key < lo_key {
                hi_key
            } else {
                lo_key
            }
        } else if hi_key < mid_key {
            if hi_key < lo_key { lo_key } else { hi_key }
        } else {
            mid_key
        }
    }

    /// `bvh_partition`: Hoare partition of `[lo, hi)` around `pivot`; returns the cut.
    fn partition(&mut self, lo: i64, hi: i64, pivot: f32, axis: usize) -> i64 {
        let (mut i, mut j) = (lo, hi);
        loop {
            while self.key(i, axis) < pivot {
                i += 1;
            }
            j -= 1;
            while pivot < self.key(j, axis) {
                j -= 1;
            }
            if i >= j {
                return i;
            }
            self.leafs.swap(i as usize, j as usize);
            i += 1;
        }
    }

    /// `bvh_insertionsort`.
    fn insertion_sort(&mut self, lo: i64, hi: i64, axis: usize) {
        for i in lo..hi {
            let mut j = i;
            let moved = self.leafs[i as usize];
            let moved_key = self.leaf_bv[moved][axis];
            while j != lo && moved_key < self.key(j - 1, axis) {
                self.leafs[j as usize] = self.leafs[(j - 1) as usize];
                j -= 1;
            }
            self.leafs[j as usize] = moved;
        }
    }

    /// `partition_nth_element`.
    fn partition_nth(&mut self, mut begin: i64, mut end: i64, nth: i64, axis: usize) {
        while end - begin > 3 {
            let pivot = self.median_of_3(begin, begin + (end - begin) / 2, end - 1, axis);
            let cut = self.partition(begin, end, pivot, axis);
            if cut <= nth {
                begin = cut;
            } else {
                end = cut;
            }
        }
        self.insertion_sort(begin, end, axis);
    }

    /// Refit of the leaves in `[begin, end)` (`refit_kdop_hull`).
    fn refit(&self, begin: i64, end: i64) -> [f32; 6] {
        let mut bounds = [
            f32::MAX,
            -f32::MAX,
            f32::MAX,
            -f32::MAX,
            f32::MAX,
            -f32::MAX,
        ];
        for &leaf in &self.leafs[begin as usize..end as usize] {
            let leaf_bounds = self.leaf_bv[leaf];
            for axis in 0..3 {
                bounds[2 * axis] = leaf_bounds[2 * axis].min(bounds[2 * axis]);
                bounds[2 * axis + 1] = leaf_bounds[2 * axis + 1].max(bounds[2 * axis + 1]);
            }
        }
        bounds
    }

    /// One branch of `non_recursive_bvh_div_nodes_task_cb`: refit, split its leaves along the largest axis
    /// into the two child ranges, then link the children.
    fn split(&mut self, j: usize, level: usize, depth: i64, first_next: usize) {
        let parent_index = j as i64 - level as i64;
        let begin = self.helper.leafs_index(depth, parent_index);
        let end = self.helper.leafs_index(depth, parent_index + 1);
        let bounds = self.refit(begin, end);
        let split_axis = largest_axis(&bounds);
        self.branches[j].bv = bounds;
        self.branches[j].main_axis = split_axis / 2;

        let first_next = first_next as i64;
        let split_nth = self
            .helper
            .leafs_index(depth + 1, (2 * j + 1) as i64 - first_next);
        if begin < end {
            self.partition_nth(begin, end, split_nth, split_axis);
        }

        let mut node_num = 0;
        for k in 0..2 {
            let child_index = (2 * j + k) as i64;
            let child_level = child_index - first_next;
            let child_begin = self.helper.leafs_index(depth + 1, child_level);
            let child_end = self.helper.leafs_index(depth + 1, child_level + 1);
            let child = match child_end - child_begin {
                count if count > 1 => NodeRef::Branch(child_index as usize),
                1 => NodeRef::Leaf(self.leafs[child_begin as usize]),
                _ => break,
            };
            self.branches[j].children[k] = Some(child);
            node_num = k + 1;
        }
        self.branches[j].node_num = node_num;
    }
}

/// `BLI_bvhtree_balance` for a binary tree. Returns the branch array, indexed from 1 (`branches[0]` unused).
/// Branch `j` has children `2j` and `2j + 1`; the root is branch 1.
fn build_branches(leaf_bv: &[[f32; 6]]) -> Vec<Branch> {
    let leafs_num = leaf_bv.len();
    let branches_num = (leafs_num as i64 - 1).max(1) as usize;
    if leafs_num == 1 {
        // A one-leaf tree still has a root branch that holds the leaf (`non_recursive_bvh_div_nodes`).
        let bounds = leaf_bv[0];
        let mut branches = vec![Branch::EMPTY; 2];
        branches[1] = Branch {
            bv: bounds,
            children: [Some(NodeRef::Leaf(0)), None],
            node_num: 1,
            main_axis: largest_axis(&bounds) / 2,
        };
        return branches;
    }
    let mut builder = TreeBuilder {
        leaf_bv,
        leafs: (0..leafs_num).collect(),
        helper: BuildHelper::new(leafs_num),
        branches: vec![Branch::EMPTY; branches_num + 1],
    };
    let mut level = 1_usize;
    let mut depth = 1_i64;
    while level <= branches_num {
        let first_next = level * 2;
        let level_stop = first_next.min(branches_num + 1);
        for j in level..level_stop {
            builder.split(j, level, depth, first_next);
        }
        level = first_next;
        depth += 1;
    }
    builder.branches
}

/// The watertight ray with Blender's per-ray state: `BVHRayCastData` bounds and `IsectRayPrecalc`.
pub(crate) struct Ray {
    origin: Vec3,
    dot_axis: [f32; 3],
    idot_axis: [f32; 3],
    index: [usize; 6],
    kx: usize,
    ky: usize,
    kz: usize,
    sx: f32,
    sy: f32,
    sz: f32,
}

impl Ray {
    /// `BLI_bvhtree_ray_cast_ex` precalc (`bvhtree_ray_cast_data_precalc`) and
    /// `isect_ray_tri_watertight_v3_precalc` (`math_geom.cc`).
    fn new(origin: Vec3, dir: Vec3) -> Self {
        let mut dot_axis = [0.0_f32; 3];
        let mut idot_axis = [0.0_f32; 3];
        let mut index = [0_usize; 6];
        for axis in 0..3 {
            dot_axis[axis] = dot(dir, AXES[axis]);
            if dot_axis[axis].abs() < f32::EPSILON {
                dot_axis[axis] = 0.0;
                // Blender keeps the sign out of this case: the index is adjusted below anyway.
                idot_axis[axis] = f32::MAX;
            } else {
                idot_axis[axis] = 1.0 / dot_axis[axis];
            }
            let low = usize::from(idot_axis[axis] < 0.0);
            index[2 * axis] = low + 2 * axis;
            index[2 * axis + 1] = (1 - low) + 2 * axis;
        }

        let kz = axis_dominant(dir);
        // Blender: kx = (kz != 2) ? kz + 1 : 0, ky likewise from kx; both are (k + 1) mod 3.
        let mut kx = (kz + 1) % 3;
        let mut ky = (kx + 1) % 3;
        // Swap kx and ky to keep the triangle winding when the ray points along -kz.
        if dir[kz] < 0.0 {
            std::mem::swap(&mut kx, &mut ky);
        }
        let inv_dir_z = 1.0 / dir[kz];
        Self {
            origin,
            dot_axis,
            idot_axis,
            index,
            kx,
            ky,
            kz,
            sx: dir[kx] * inv_dir_z,
            sy: dir[ky] * inv_dir_z,
            sz: inv_dir_z,
        }
    }

    /// `isect_ray_tri_watertight_v3` (`math_geom.cc`): the ray parameter of the hit, which may face either way.
    pub(crate) fn watertight(&self, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
        let a = sub(v0, self.origin);
        let b = sub(v1, self.origin);
        let c = sub(v2, self.origin);
        let ax = a[self.kx] - self.sx * a[self.kz];
        let ay = a[self.ky] - self.sy * a[self.kz];
        let bx = b[self.kx] - self.sx * b[self.kz];
        let by = b[self.ky] - self.sy * b[self.kz];
        let cx = c[self.kx] - self.sx * c[self.kz];
        let cy = c[self.ky] - self.sy * c[self.kz];
        let u = cx * by - cy * bx;
        let v = ax * cy - ay * cx;
        let w = bx * ay - by * ax;
        if (u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0) {
            return None;
        }
        let det = u + v + w;
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let t = (u * a[self.kz] + v * b[self.kz] + w * c[self.kz]) * self.sz;
        let sign_t = if det.is_sign_negative() { -t } else { t };
        if sign_t < 0.0 {
            return None;
        }
        Some(t * (1.0 / det))
    }

    /// `fast_ray_nearest_hit` (`BLI_kdopbvh.cc`): the entry distance of a box, or `FLT_MAX` for a miss. The
    /// box distances are absolute; `hit_dist` is whatever the callback last stored.
    fn fast_nearest_hit(&self, bv: &[f32; 6], hit_dist: f32) -> f32 {
        let o = self.origin;
        let idot = self.idot_axis;
        let ix = self.index;
        let t1x = (bv[ix[0]] - o[0]) * idot[0];
        let t2x = (bv[ix[1]] - o[0]) * idot[0];
        let t1y = (bv[ix[2]] - o[1]) * idot[1];
        let t2y = (bv[ix[3]] - o[1]) * idot[1];
        let t1z = (bv[ix[4]] - o[2]) * idot[2];
        let t2z = (bv[ix[5]] - o[2]) * idot[2];
        if (t1x > t2y || t2x < t1y || t1x > t2z || t2x < t1z || t1y > t2z || t2y < t1z)
            || (t2x < 0.0 || t2y < 0.0 || t2z < 0.0)
            || (t1x > hit_dist || t1y > hit_dist || t1z > hit_dist)
        {
            return f32::MAX;
        }
        t1x.max(t1y).max(t1z)
    }
}

/// `BVHRayCastData::hit`: the leaf index (or -1) and the distance the tree prunes against.
pub(crate) struct RayHit {
    pub(crate) index: isize,
    pub(crate) dist: f32,
}

/// A binary k-DOP tree over leaf boxes, balanced as `BLI_bvhtree_balance` does. Leaf `i` is the
/// `i`-th box passed to `from_leaf_bounds`; the caller maps leaf indices to its own primitives.
pub(crate) struct KdopTree {
    leaf_bv: Vec<[f32; 6]>,
    branches: Vec<Branch>,
    root: Option<NodeRef>,
}

impl KdopTree {
    /// `BLI_bvhtree_balance` over leaf boxes in insertion order. An empty list gives no root.
    pub(crate) fn from_leaf_bounds(leaf_bv: &[[f32; 6]]) -> Self {
        let (branches, root) = if leaf_bv.is_empty() {
            (Vec::new(), None)
        } else {
            (build_branches(leaf_bv), Some(NodeRef::Branch(1)))
        };
        Self {
            leaf_bv: leaf_bv.to_vec(),
            branches,
            root,
        }
    }

    /// The leaf box of one triangle: its extent inflated by the tree epsilon (`create_kdop_hull`).
    pub(crate) fn leaf_bounds_for_triangle(tri: [Vec3; 3]) -> [f32; 6] {
        triangle_bounds(tri)
    }

    /// `BLI_bvhtree_ray_cast_ex` with `BVH_RAYCAST_WATERTIGHT`. `callback` receives each leaf index the
    /// traversal reaches, the ray, and the running hit, and stores hits in `hit`. The caller sets the
    /// initial `hit` (`index: -1`, `dist: f32::MAX` for an open cast), as `BLI_bvhtree_ray_cast`'s
    /// `hit` argument does. Returns the index of the hit leaf, or -1.
    pub(crate) fn ray_cast<F>(
        &self,
        origin: Vec3,
        dir: Vec3,
        hit: &mut RayHit,
        callback: &mut F,
    ) -> isize
    where
        F: FnMut(usize, &Ray, &mut RayHit),
    {
        let ray = Ray::new(origin, dir);
        if let Some(root) = self.root {
            self.dfs(&ray, root, hit, callback);
        }
        hit.index
    }

    /// `dfs_raycast`: prune by the node box, visit leaves through the callback, and descend branches in the
    /// order `dot_axis[main_axis]` gives.
    fn dfs<F>(&self, ray: &Ray, node: NodeRef, hit: &mut RayHit, callback: &mut F)
    where
        F: FnMut(usize, &Ray, &mut RayHit),
    {
        let bv = match node {
            NodeRef::Leaf(leaf) => self.leaf_bv[leaf],
            NodeRef::Branch(branch) => self.branches[branch].bv,
        };
        let dist = ray.fast_nearest_hit(&bv, hit.dist);
        if dist >= hit.dist {
            return;
        }
        match node {
            NodeRef::Leaf(leaf) => callback(leaf, ray, hit),
            NodeRef::Branch(branch) => {
                let branch = self.branches[branch];
                let children = branch
                    .children
                    .iter()
                    .take(branch.node_num)
                    .flatten()
                    .copied();
                if ray.dot_axis[branch.main_axis] > 0.0 {
                    for child in children {
                        self.dfs(ray, child, hit, callback);
                    }
                } else {
                    for child in children.rev() {
                        self.dfs(ray, child, hit, callback);
                    }
                }
            }
        }
    }

    /// `BLI_bvhtree_find_nearest_ex`: the leaf nearest to `co`, searched from `nearest` (its `dist_sq`
    /// bounds the search; a `dist_sq` of `f32::MAX` is an open search). `callback` receives the leaf
    /// index, the query point and the running nearest, and writes the leaf's nearest point into it.
    /// `optimal_order` selects best-first search with a heap (`BVH_NEAREST_OPTIMAL_ORDER`) instead of
    /// depth-first search. Returns the index of the nearest leaf, or -1.
    pub(crate) fn find_nearest<F>(
        &self,
        co: Vec3,
        nearest: &mut NearestHit,
        callback: &mut F,
        optimal_order: bool,
    ) -> isize
    where
        F: FnMut(usize, Vec3, &mut NearestHit),
    {
        if let Some(root) = self.root {
            let (root_dist_sq, _) = self.nearest_box(co, root);
            if root_dist_sq < nearest.dist_sq {
                if optimal_order {
                    let mut heap = HeapSimple::new();
                    self.heap_nearest_expand(co, root, &mut heap, nearest, callback);
                    while !heap.is_empty() && heap.top_value() < nearest.dist_sq {
                        let Some(node) = heap.pop_min() else {
                            break;
                        };
                        self.heap_nearest_expand(co, node, &mut heap, nearest, callback);
                    }
                } else {
                    self.dfs_nearest(co, root, nearest, callback);
                }
            }
        }
        nearest.index
    }

    /// `dfs_find_nearest_dfs`: prune children by their box distance, dive toward the query side of
    /// each branch's split axis, and let the callback decide leaves.
    fn dfs_nearest<F>(&self, co: Vec3, node: NodeRef, nearest: &mut NearestHit, callback: &mut F)
    where
        F: FnMut(usize, Vec3, &mut NearestHit),
    {
        match node {
            NodeRef::Leaf(leaf) => callback(leaf, co, nearest),
            NodeRef::Branch(branch) => {
                let branch = self.branches[branch];
                let children: Vec<NodeRef> = branch
                    .children
                    .iter()
                    .take(branch.node_num)
                    .flatten()
                    .copied()
                    .collect();
                let Some(&first) = children.first() else {
                    return;
                };
                let forward =
                    co[branch.main_axis] <= self.node_bounds(first)[branch.main_axis * 2 + 1];
                if forward {
                    for &child in &children {
                        if self.nearest_box(co, child).0 >= nearest.dist_sq {
                            continue;
                        }
                        self.dfs_nearest(co, child, nearest, callback);
                    }
                } else {
                    for &child in children.iter().rev() {
                        if self.nearest_box(co, child).0 >= nearest.dist_sq {
                            continue;
                        }
                        self.dfs_nearest(co, child, nearest, callback);
                    }
                }
            }
        }
    }

    /// `heap_find_nearest_inner`: a leaf goes to the callback; a branch queues each child closer than
    /// the current nearest, keyed by its box distance.
    fn heap_nearest_expand<F>(
        &self,
        co: Vec3,
        node: NodeRef,
        heap: &mut HeapSimple<NodeRef>,
        nearest: &mut NearestHit,
        callback: &mut F,
    ) where
        F: FnMut(usize, Vec3, &mut NearestHit),
    {
        match node {
            NodeRef::Leaf(leaf) => callback(leaf, co, nearest),
            NodeRef::Branch(branch) => {
                let branch = self.branches[branch];
                for &child in branch.children.iter().take(branch.node_num).flatten() {
                    let dist_sq = self.nearest_box(co, child).0;
                    if dist_sq < nearest.dist_sq {
                        heap.insert(dist_sq, child);
                    }
                }
            }
        }
    }

    /// The bounds of a node: a leaf's box, or a branch's box over its leaves.
    fn node_bounds(&self, node: NodeRef) -> [f32; 6] {
        match node {
            NodeRef::Leaf(leaf) => self.leaf_bv[leaf],
            NodeRef::Branch(branch) => self.branches[branch].bv,
        }
    }

    /// `calc_nearest_point_squared`: the squared distance from `co` to the node's box, and the box point
    /// nearest to `co`. Each axis is clamped to the box slab, lower bound first, as Blender does.
    fn nearest_box(&self, co: Vec3, node: NodeRef) -> (f32, Vec3) {
        let bv = self.node_bounds(node);
        let mut nearest = [0.0_f32; 3];
        for axis in 0..3 {
            let value = co[axis].max(bv[2 * axis]).min(bv[2 * axis + 1]);
            nearest[axis] = value;
        }
        let offset = sub(co, nearest);
        (dot(offset, offset), nearest)
    }
}

/// Blender's `BVHTreeNearest`: the nearest leaf found so far, its nearest point, and the squared
/// distance to it. `index` is -1 when no leaf has been accepted.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NearestHit {
    pub(crate) index: isize,
    pub(crate) co: Vec3,
    pub(crate) dist_sq: f32,
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "small tree fixtures")]

    use super::*;

    /// Leaf boxes of `count` triangles spread over uneven extents, so the balance splits on every axis.
    fn leaves(count: usize) -> Vec<[f32; 6]> {
        (0..count)
            .map(|index| {
                let step = index as f32;
                KdopTree::leaf_bounds_for_triangle([
                    [step * 0.5, (step * 7.0) % 5.0, step],
                    [step * 0.5 + 1.0, (step * 3.0) % 4.0, step + 0.25],
                    [step * 0.25, (step * 11.0) % 6.0 + 2.0, -step],
                ])
            })
            .collect()
    }

    #[test]
    fn every_leaf_is_reachable_from_the_root() {
        let tree = KdopTree::from_leaf_bounds(&leaves(37));
        let mut seen = [false; 37];
        let mut stack = vec![tree.root.expect("tree has leaves")];
        while let Some(node) = stack.pop() {
            match node {
                NodeRef::Leaf(leaf) => seen[leaf] = true,
                NodeRef::Branch(branch) => {
                    let branch = tree.branches[branch];
                    stack.extend(
                        branch
                            .children
                            .iter()
                            .take(branch.node_num)
                            .flatten()
                            .copied(),
                    );
                }
            }
        }
        assert!(
            seen.iter().all(|&s| s),
            "every triangle is a leaf of the tree"
        );
    }
}
