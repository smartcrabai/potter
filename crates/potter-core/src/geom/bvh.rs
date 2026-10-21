//! Blender's k-DOP bounding-volume tree, ported in `f32` from `BLI_kdopbvh.cc` (pinned 5.2.2). This is
//! the one BVH of the codebase:
//!
//! - `BLI_bvhtree_new(n, epsilon, tree_type, axis)` plus `BLI_bvhtree_insert` is [`kdop_bounds`] over
//!   each leaf, then [`KdopTree::new`] (`BLI_bvhtree_balance`, `non_recursive_bvh_div_nodes`).
//! - `BLI_bvhtree_ray_cast_ex` with zero radius is [`KdopTree::ray_cast`] (`dfs_raycast`).
//! - `BLI_bvhtree_overlap` is [`overlap_pairs`] (`tree_overlap_traverse`).
//!
//! `tree_type` is the number of children per branch, and `axis` the number of bound entries per box:
//! 6 for the three slabs `x`, `y`, `z`, or 8 for those plus `x + y + z`. The ray test reads the first
//! three slabs, and the overlap test all `axis / 2` slabs, as Blender does. Leaf `i` is the `i`-th box
//! given to [`KdopTree::new`], and the caller maps leaf indices to its own primitives.
//!
//! The Watertight ray ([`Ray::watertight`]) is `isect_ray_tri_watertight_v3`, used by the cage ray cast.

/// A point or direction in `f32`, as Blender stores them.
pub(crate) type Vec3 = [f32; 3];

/// Slab bounds of one box, `[min, max]` per slab. A 6-axis tree reads the first three slabs, an 8-axis
/// tree four. Unused entries are ignored.
pub(crate) type Bounds = [f32; 8];

/// Blender `bvhtree_kdop_axes`: `x`, `y`, `z`, and `x + y + z` (the last one unnormalised).
const AXES: [Vec3; 4] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
];
/// Blender `bvhtree_kdop_axes_length`: each slab's leaf inflation is `epsilon * length`.
const AXES_LENGTH: [f32; 4] = [1.0, 1.0, 1.0, 1.732_050_8];
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
/// Blender `BVH_RAYCAST_DIST_MAX` (`BLI_kdopbvh.hh`).
const RAYCAST_DIST_MAX: f32 = f32::MAX / 2.0;

pub(crate) fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: Vec3, factor: f32) -> Vec3 {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

pub(crate) fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `normalize_v3_v3`: the unit vector and its length. A near-zero vector stays unchanged with length 0.
pub(crate) fn normalize(v: Vec3) -> (Vec3, f32) {
    let squared = dot(v, v);
    if squared > 1.0e-35 {
        let length = squared.sqrt();
        (scale(v, 1.0 / length), length)
    } else {
        (v, 0.0)
    }
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

/// The box of `points` over `axis` entries (`create_kdop_hull`), inflated by `epsilon` on every slab
/// (`bvhtree_node_inflate`). Blender's `BLI_bvhtree_new` clamps the epsilon to `FLT_EPSILON`.
pub(crate) fn kdop_bounds(points: &[Vec3], epsilon: f32, axis: usize) -> Bounds {
    let slabs = axis / 2;
    let mut bounds = EMPTY_BOUNDS;
    for &point in points {
        for (slab, &direction) in AXES.iter().take(slabs).enumerate() {
            let d = dot(point, direction);
            bounds[2 * slab] = d.min(bounds[2 * slab]);
            bounds[2 * slab + 1] = d.max(bounds[2 * slab + 1]);
        }
    }
    let epsilon = epsilon.max(f32::EPSILON);
    for (slab, &length) in AXES_LENGTH.iter().take(slabs).enumerate() {
        let dist = epsilon * length;
        bounds[2 * slab] -= dist;
        bounds[2 * slab + 1] += dist;
    }
    bounds
}

/// The union of the boxes of `ids` over `slabs` slabs (`refit_kdop_hull`).
fn refit(leaves: &[Bounds], ids: &[usize], slabs: usize) -> Bounds {
    let mut bounds = EMPTY_BOUNDS;
    for &id in ids {
        for slab in 0..slabs {
            bounds[2 * slab] = leaves[id][2 * slab].min(bounds[2 * slab]);
            bounds[2 * slab + 1] = leaves[id][2 * slab + 1].max(bounds[2 * slab + 1]);
        }
    }
    bounds
}

/// `get_largest_axis` (`BLI_kdopbvh.cc`): the bound index of the largest extent (1, 3 or 5, i.e. the
/// axis's maximum slot). Ties go to the later axis, as in Blender.
fn largest_axis(bv: &Bounds) -> usize {
    let (m0, m1, m2) = (bv[1] - bv[0], bv[3] - bv[2], bv[5] - bv[4]);
    if m0 > m1 {
        if m0 > m2 { 1 } else { 5 }
    } else if m1 > m2 {
        3
    } else {
        5
    }
}

/// A node link: a leaf (box index) or a branch (index into the branch array).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Child {
    Leaf(usize),
    Branch(usize),
}

/// Blender `BVHNode` of a branch: its box over every leaf below it, its split slab, and its children.
#[derive(Clone, Default)]
struct Branch {
    bounds: Bounds,
    /// Blender `main_axis`: the slab (0, 1, 2) of the split axis.
    main_axis: usize,
    children: Vec<Child>,
}

/// Blender `BVHBuildHelper`: the layout of `leafs_num` leaves in an implicit `tree_type`-ary tree.
struct Implicit {
    leafs_num: usize,
    leafs_per_child: [usize; 32],
    branches_on_level: [usize; 32],
    remain_leafs: usize,
}

impl Implicit {
    /// Blender `build_implicit_tree_helper`.
    fn new(leafs_num: usize, tree_type: usize) -> Self {
        let mut leafs_per_child = [0usize; 32];
        let mut branches_on_level = [0usize; 32];
        leafs_per_child[0] = 1;
        while leafs_per_child[0] < leafs_num {
            leafs_per_child[0] *= tree_type;
        }
        branches_on_level[0] = 1;
        let mut depth = 1;
        while depth < 32 && leafs_per_child[depth - 1] != 0 {
            branches_on_level[depth] = branches_on_level[depth - 1] * tree_type;
            leafs_per_child[depth] = leafs_per_child[depth - 1] / tree_type;
            depth += 1;
        }
        let remain = leafs_num - leafs_per_child[1];
        let nnodes = (remain + tree_type - 2) / (tree_type - 1);
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
fn child_index(parent: usize, k: usize, tree_type: usize) -> usize {
    parent * tree_type + k + 2 - tree_type
}

/// Blender `bvh_medianof3`: the median of three leaves by their key on `axis`.
fn median_of3(a: &[usize], lo: usize, mid: usize, hi: usize, axis: usize, leaves: &[Bounds]) -> usize {
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
fn partition(a: &mut [usize], lo: usize, hi: usize, pivot_key: f32, axis: usize, leaves: &[Bounds]) -> usize {
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
fn partition_nth_element(a: &mut [usize], mut begin: usize, mut end: usize, n: usize, axis: usize, leaves: &[Bounds]) {
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
fn split_leafs(order: &mut [usize], nth: &[usize], axis: usize, leaves: &[Bounds], tree_type: usize) {
    for i in 0..tree_type - 1 {
        if nth[i] >= nth[tree_type] {
            break;
        }
        partition_nth_element(order, nth[i], nth[tree_type], nth[i + 1], axis, leaves);
    }
}

/// Blender `non_recursive_bvh_div_nodes_task_cb`: fits branch `j`, splits its leaves along its largest
/// axis, and links its children.
#[expect(clippy::too_many_arguments, reason = "mirrors the Blender helper's state, one value each")]
fn divide(
    branches: &mut [Branch],
    order: &mut [usize],
    leaves: &[Bounds],
    implicit: &Implicit,
    level: Level,
    j: usize,
    tree_type: usize,
    axis: usize,
) {
    let parent_level_index = j - level.start;
    let begin = implicit.leafs_index(level.depth, parent_level_index);
    let end = implicit.leafs_index(level.depth, parent_level_index + 1);
    let bounds = refit(leaves, &order[begin..end], axis / 2);
    let split_axis = largest_axis(&bounds);

    let mut nth = vec![0usize; tree_type + 1];
    nth[0] = begin;
    nth[tree_type] = end;
    for (k, slot) in nth.iter_mut().enumerate().take(tree_type).skip(1) {
        let child_level_index = child_index(j, k, tree_type) - level.next;
        *slot = implicit.leafs_index(level.depth + 1, child_level_index);
    }
    split_leafs(order, &nth, split_axis, leaves, tree_type);

    let mut children = Vec::with_capacity(tree_type);
    for k in 0..tree_type {
        let index = child_index(j, k, tree_type);
        let child_level_index = index - level.next;
        let first = implicit.leafs_index(level.depth + 1, child_level_index);
        let last = implicit.leafs_index(level.depth + 1, child_level_index + 1);
        if last > first + 1 {
            children.push(Child::Branch(index));
        } else if last == first + 1 {
            children.push(Child::Leaf(order[first]));
        } else {
            break;
        }
    }
    branches[j] = Branch {
        bounds,
        main_axis: split_axis / 2,
        children,
    };
}

/// Blender `BLI_bvhtree_balance` (`non_recursive_bvh_div_nodes`) over the leaf boxes, in insertion
/// order. Returns the branches indexed from 1: branch `j` is Blender's `tree[j]`, and the root is 1.
fn build_branches(leaves: &[Bounds], tree_type: usize, axis: usize) -> Vec<Branch> {
    let leafs_num = leaves.len();
    let branches_num = ((leafs_num + tree_type - 3) / (tree_type - 1)).max(1);
    let mut branches = vec![Branch::default(); branches_num + 1];
    if leafs_num == 1 {
        // A one-leaf tree still has a root branch that holds the leaf.
        branches[1] = Branch {
            bounds: leaves[0],
            main_axis: largest_axis(&leaves[0]) / 2,
            children: vec![Child::Leaf(0)],
        };
        return branches;
    }
    let implicit = Implicit::new(leafs_num, tree_type);
    let mut order: Vec<usize> = (0..leafs_num).collect();
    let mut start = 1;
    let mut depth = 1;
    while start <= branches_num {
        let next = child_index(start, 0, tree_type);
        let stop = next.min(branches_num + 1);
        let level = Level { start, next, depth };
        for j in start..stop {
            divide(&mut branches, &mut order, leaves, &implicit, level, j, tree_type, axis);
        }
        start = next;
        depth += 1;
    }
    branches
}

/// A balanced k-DOP tree over leaf boxes. Leaf `i` is the `i`-th box given to [`KdopTree::new`].
pub(crate) struct KdopTree {
    /// Blender `axis`: the bound entries per box, 6 or 8.
    axis: usize,
    leaves: Vec<Bounds>,
    /// Branch `j` is `branches[j]`, Blender's implicit numbering. Empty when there are no leaves.
    branches: Vec<Branch>,
}

impl KdopTree {
    /// `BLI_bvhtree_balance` over `leaves` for `BLI_bvhtree_new(n, epsilon, tree_type, axis)`. The
    /// leaves are the boxes [`kdop_bounds`] gives for the same `epsilon` and `axis`.
    ///
    /// Cost: O(T log T) time and O(T) memory for T leaves.
    pub(crate) fn new(leaves: Vec<Bounds>, tree_type: usize, axis: usize) -> Self {
        let branches = if leaves.is_empty() {
            Vec::new()
        } else {
            build_branches(&leaves, tree_type, axis)
        };
        Self {
            axis,
            leaves,
            branches,
        }
    }

    /// The leaves reachable from the root, in depth-first order (every leaf once for a valid tree).
    #[cfg(test)]
    pub(crate) fn reachable_leaves(&self) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = Vec::new();
        if !self.branches.is_empty() {
            stack.push(Child::Branch(1));
        }
        while let Some(child) = stack.pop() {
            match child {
                Child::Leaf(leaf) => out.push(leaf),
                Child::Branch(branch) => stack.extend(self.branches[branch].children.iter().rev().copied()),
            }
        }
        out
    }

    fn child_bounds(&self, child: Child) -> Bounds {
        match child {
            Child::Leaf(leaf) => self.leaves[leaf],
            Child::Branch(branch) => self.branches[branch].bounds,
        }
    }

    /// `BLI_bvhtree_ray_cast_ex` with zero radius and `BVH_RAYCAST_WATERTIGHT`: `dfs_raycast` from the root.
    /// `callback` is called for each leaf whose box can still hold a closer hit (`dist < hit.dist`), with
    /// the ray, and may update `hit`. The walk stops descending into boxes behind the ray or beyond
    /// `hit.dist`. Returns the index of the hit leaf, or -1.
    ///
    /// `origin` and `dir` are as Blender passes them. `dir` must be nonzero in every component for the
    /// slab tests, which the callers ensure.
    pub(crate) fn ray_cast<F>(&self, origin: Vec3, dir: Vec3, hit: &mut RayHit, callback: &mut F) -> isize
    where
        F: FnMut(usize, &Ray, &mut RayHit),
    {
        if self.branches.is_empty() {
            return hit.index;
        }
        let ray = Ray::new(origin, dir);
        self.dfs(&ray, Child::Branch(1), hit, callback);
        hit.index
    }

    fn dfs<F>(&self, ray: &Ray, child: Child, hit: &mut RayHit, callback: &mut F)
    where
        F: FnMut(usize, &Ray, &mut RayHit),
    {
        let bounds = match child {
            Child::Leaf(leaf) => self.leaves[leaf],
            Child::Branch(branch) => self.branches[branch].bounds,
        };
        let dist = ray.fast_nearest_hit(&bounds, hit.dist);
        if dist >= hit.dist {
            return;
        }
        match child {
            Child::Leaf(leaf) => callback(leaf, ray, hit),
            Child::Branch(branch) => {
                let branch = &self.branches[branch];
                // Dive in the direction of the ray along the split axis.
                if ray.dot_axis[branch.main_axis] > 0.0 {
                    for &next in &branch.children {
                        self.dfs(ray, next, hit, callback);
                    }
                } else {
                    for &next in branch.children.iter().rev() {
                        self.dfs(ray, next, hit, callback);
                    }
                }
            }
        }
    }
}

/// `BLI_bvhtree_overlap_ex` over two trees: `tree_overlap_traverse` from the roots, descending both
/// trees and pairing leaves whose boxes overlap on every slab of the first tree's `axis`. Each pair is
/// `(leaf of first, leaf of second)`. Blender's `BLI_stack_pop_n` hands the pushed pairs back last-first,
/// so the result is reversed.
pub(crate) fn overlap_pairs(first: &KdopTree, second: &KdopTree) -> Vec<(usize, usize)> {
    if first.branches.is_empty() || second.branches.is_empty() {
        return Vec::new();
    }
    let slabs = first.axis.min(second.axis) / 2;
    let mut out = Vec::new();
    overlap_traverse(first, Child::Branch(1), second, Child::Branch(1), slabs, &mut out);
    out.reverse();
    out
}

/// Blender `tree_overlap_traverse`: while the boxes overlap, descend the first node while it is a branch,
/// otherwise the second node's children, and emit a pair when both are leaves.
fn overlap_traverse(
    tree1: &KdopTree,
    node1: Child,
    tree2: &KdopTree,
    node2: Child,
    slabs: usize,
    out: &mut Vec<(usize, usize)>,
) {
    let a = tree1.child_bounds(node1);
    let b = tree2.child_bounds(node2);
    if !(0..slabs).all(|slab| a[2 * slab] <= b[2 * slab + 1] && b[2 * slab] <= a[2 * slab + 1]) {
        return;
    }
    match (node1, node2) {
        (Child::Leaf(id1), Child::Leaf(id2)) => out.push((id1, id2)),
        (Child::Leaf(_), Child::Branch(index)) => {
            for &child in &tree2.branches[index].children {
                overlap_traverse(tree1, node1, tree2, child, slabs, out);
            }
        }
        (Child::Branch(index), _) => {
            for &child in &tree1.branches[index].children {
                overlap_traverse(tree1, child, tree2, node2, slabs, out);
            }
        }
    }
}

/// Blender `BVHRayCastData::hit`: the leaf index (or -1) and the distance the tree prunes against.
pub(crate) struct RayHit {
    pub(crate) index: isize,
    pub(crate) dist: f32,
}

impl RayHit {
    /// The open cast start: no leaf, distance `BVH_RAYCAST_DIST_MAX`.
    pub(crate) fn open() -> Self {
        Self {
            index: -1,
            dist: RAYCAST_DIST_MAX,
        }
    }

    /// The hit leaf, if a callback has accepted one.
    pub(crate) fn leaf(&self) -> Option<usize> {
        usize::try_from(self.index).ok()
    }
}

/// The watertight ray with Blender's per-ray state: `BVHRayCastData` bounds and `IsectRayPrecalc`.
pub(crate) struct Ray {
    origin: Vec3,
    pub(crate) dir: Vec3,
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
            dir,
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
    fn fast_nearest_hit(&self, bv: &Bounds, hit_dist: f32) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::{Bounds, KdopTree, RayHit, kdop_bounds, overlap_pairs};

    /// A small deterministic generator, so the fixtures do not depend on a library RNG.
    struct Lcg(u64);

    impl Lcg {
        fn unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32) / ((1_u64 << 24) as f32)
        }

        fn point(&mut self) -> [f32; 3] {
            [self.unit(), self.unit(), self.unit()]
        }
    }

    /// Boxes of `count` random triangles inside the unit cube, at the given tree epsilon and axis count.
    fn random_leaves(rng: &mut Lcg, count: usize, epsilon: f32, axis: usize) -> Vec<Bounds> {
        (0..count)
            .map(|_| {
                let points = [rng.point(), rng.point(), rng.point()];
                kdop_bounds(&points, epsilon, axis)
            })
            .collect()
    }

    #[test]
    fn every_leaf_is_reachable_from_the_root_for_every_tree_type() {
        for (tree_type, axis) in [(2, 6), (4, 6), (8, 8)] {
            for count in [1_usize, 2, 3, 37] {
                let mut rng = Lcg(7 + count as u64);
                let leaves = random_leaves(&mut rng, count, 1.0e-4, axis);
                let tree = KdopTree::new(leaves, tree_type, axis);
                let mut seen = vec![false; count];
                let mut stack = vec![super::Child::Branch(1)];
                while let Some(node) = stack.pop() {
                    match node {
                        super::Child::Leaf(leaf) => seen[leaf] = true,
                        super::Child::Branch(branch) => stack.extend(tree.branches[branch].children.iter().copied()),
                    }
                }
                assert!(seen.iter().all(|&s| s), "tree_type {tree_type}, {count} leaves");
            }
        }
    }

    #[test]
    fn ray_cast_visits_every_leaf_the_ray_crosses() {
        let mut rng = Lcg(11);
        let leaves = random_leaves(&mut rng, 60, 1.0e-4, 8);
        let tree = KdopTree::new(leaves.clone(), 8, 8);
        for _ in 0..200 {
            let origin = [rng.unit() * 2.0 - 0.5, rng.unit() * 2.0 - 0.5, rng.unit() * 2.0 - 0.5];
            let dir = [rng.unit() * 2.0 - 1.0, rng.unit() * 2.0 - 1.0, rng.unit() * 2.0 - 1.0];
            let mut visited = Vec::new();
            let mut hit = RayHit::open();
            tree.ray_cast(origin, dir, &mut hit, &mut |leaf: usize, _ray: &super::Ray, _hit: &mut RayHit| {
                visited.push(leaf);
            });
            // Every leaf whose box the ray crosses with a margin is visited, and no leaf that misses by a margin.
            for (leaf, bounds) in leaves.iter().enumerate() {
                let crossed = |grow: f32| {
                    let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
                    for axis in 0..3 {
                        let (min, max) = (f64::from(bounds[2 * axis]) - f64::from(grow), f64::from(bounds[2 * axis + 1]) + f64::from(grow));
                        let (o, d) = (f64::from(origin[axis]), f64::from(dir[axis]));
                        let (t1, t2) = ((min - o) / d, (max - o) / d);
                        lo = lo.max(t1.min(t2));
                        hi = hi.min(t1.max(t2));
                    }
                    lo <= hi && hi >= 0.0
                };
                let strict = crossed(-1.0e-5);
                let loose = crossed(1.0e-5);
                if strict {
                    assert!(visited.contains(&leaf), "leaf {leaf} is crossed but not visited");
                }
                if !loose {
                    assert!(!visited.contains(&leaf), "leaf {leaf} is not crossed but visited");
                }
            }
        }
    }

    #[test]
    fn overlap_pairs_match_the_box_overlap_set() {
        for (tree_type, axis) in [(8, 8), (2, 6)] {
            let mut rng = Lcg(23);
            let first = random_leaves(&mut rng, 40, 1.0e-4, axis);
            let second = random_leaves(&mut rng, 30, 1.0e-4, axis);
            let slabs = axis / 2;
            let mut expected: Vec<(usize, usize)> = Vec::new();
            for (a, fa) in first.iter().enumerate() {
                for (b, sb) in second.iter().enumerate() {
                    if (0..slabs).all(|s| fa[2 * s] <= sb[2 * s + 1] && sb[2 * s] <= fa[2 * s + 1]) {
                        expected.push((a, b));
                    }
                }
            }
            let mut got = overlap_pairs(&KdopTree::new(first, tree_type, axis), &KdopTree::new(second, tree_type, axis));
            got.sort_unstable();
            expected.sort_unstable();
            assert_eq!(got, expected, "tree_type {tree_type}, axis {axis}");
        }
    }
}
