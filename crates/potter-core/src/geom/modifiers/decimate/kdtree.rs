//! Balanced 3D KD-tree for `float3` points, mirroring Blender's `BLI_kdtree.hh`.

/// Sentinel link for an absent child (`kd_node_unset`).
const UNSET: u32 = u32::MAX;

/// Capacity of the range-search stack. A median-balanced tree over `u32` links has at most 33
/// levels and the stack holds at most one pending sibling per ancestor plus two fresh children.
const STACK_LEN: usize = 64;

struct Node {
    co: [f32; 3],
    index: u32,
    /// Split axis (`KDTreeNode::d`). Only set on subtree medians; leaves keep 0 as in BLI.
    d: u8,
    left: u32,
    right: u32,
}

/// Mirrors `KDTree<float3>` built by `kdtree_new` + `kdtree_insert` + `kdtree_balance`.
pub(super) struct KdTree {
    nodes: Vec<Node>,
    root: u32,
}

struct Stack {
    items: [u32; STACK_LEN],
    len: usize,
}

impl Stack {
    fn push(&mut self, link: u32) {
        self.items[self.len] = link;
        self.len += 1;
    }

    fn pop(&mut self) -> u32 {
        self.len -= 1;
        self.items[self.len]
    }
}

impl KdTree {
    /// Mirrors `kdtree_new<float3>(n)`, `kdtree_insert(tree, i, points[i])` in index order, then
    /// `kdtree_balance`. Point `i` keeps index `i`.
    pub(super) fn new(points: &[[f32; 3]]) -> KdTree {
        assert!(
            points.len() < UNSET as usize,
            "KdTree holds fewer than u32::MAX points"
        );
        let mut nodes: Vec<Node> = points
            .iter()
            .enumerate()
            .map(|(i, &co)| Node {
                co,
                index: i as u32,
                d: 0,
                left: UNSET,
                right: UNSET,
            })
            .collect();
        let root = balance(&mut nodes, 0, 0);
        KdTree { nodes, root }
    }

    /// Mirrors `kdtree_range_search_cb<float3>`.
    ///
    /// Stop rule (as in BLI's `if (search_cb(...) == false) break;`): `visit` returns `true` to
    /// keep searching and `false` to abort the whole search immediately. It is called only for
    /// points with `dist_sq <= radius * radius`, in pre-order (node, then right subtree, then
    /// left subtree), with `dist_sq` the squared distance. The order is not sorted by distance.
    pub(super) fn range_search(
        &self,
        co: [f32; 3],
        radius: f32,
        mut visit: impl FnMut(usize, [f32; 3], f32) -> bool,
    ) {
        if self.root == UNSET {
            return;
        }
        let range_sq = radius * radius;
        let mut stack = Stack {
            items: [0; STACK_LEN],
            len: 0,
        };
        stack.push(self.root);
        while stack.len > 0 {
            let node = &self.nodes[stack.pop() as usize];
            let axis = node.d as usize;
            if co[axis] + radius < node.co[axis] {
                if node.left != UNSET {
                    stack.push(node.left);
                }
            } else if co[axis] - radius > node.co[axis] {
                if node.right != UNSET {
                    stack.push(node.right);
                }
            } else {
                let dist_sq = distance_squared(node.co, co);
                if dist_sq <= range_sq && !visit(node.index as usize, node.co, dist_sq) {
                    return;
                }
                if node.left != UNSET {
                    stack.push(node.left);
                }
                if node.right != UNSET {
                    stack.push(node.right);
                }
            }
        }
    }
}

/// Mirrors `math::distance_squared` for `float3`: `dot(a - b, a - b)` summed left to right.
fn distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let mut result = d[0] * d[0];
    result += d[1] * d[1];
    result += d[2] * d[2];
    result
}

/// Mirrors `detail::kdtree_balance`. `ofs` is the global index of `nodes[0]`; returns the global
/// index of the subtree root, or `UNSET` when empty.
#[expect(
    clippy::expect_used,
    reason = "median is len / 2, which is below len for len >= 2"
)]
fn balance(nodes: &mut [Node], axis: usize, ofs: u32) -> u32 {
    match nodes.len() {
        0 => UNSET,
        1 => ofs,
        len => {
            let median = len / 2;
            select_median(nodes, axis, median);
            let (left_part, rest) = nodes.split_at_mut(median);
            let (node, right_part) = rest.split_first_mut().expect("median < len");
            node.d = axis as u8;
            let next_axis = (axis + 1) % 3;
            node.left = balance(left_part, next_axis, ofs);
            node.right = balance(right_part, next_axis, ofs + median as u32 + 1);
            ofs + median as u32
        }
    }
}

/// Quick-sort style partition from `detail::kdtree_balance`: pivot is the last element of the
/// active range, equal keys stop both scans, and the range shrinks towards `median`. Indices are
/// signed so the `i = left - 1` start and `right = i - 1` updates need no wrapping.
#[expect(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "the scan stops on !(a < b), so NaN keys stop it exactly as in BLI_kdtree"
)]
fn select_median(nodes: &mut [Node], axis: usize, median: usize) {
    let median = median as isize;
    let mut left: isize = 0;
    let mut right: isize = nodes.len() as isize - 1;
    while right > left {
        let pivot = nodes[right as usize].co[axis];
        let mut i = left - 1;
        let mut j = right;
        loop {
            i += 1;
            if !(nodes[i as usize].co[axis] < pivot) {
                break;
            }
        }
        loop {
            j -= 1;
            if !(nodes[j as usize].co[axis] > pivot && j > left) {
                break;
            }
        }
        while i < j {
            nodes.swap(i as usize, j as usize);
            loop {
                i += 1;
                if !(nodes[i as usize].co[axis] < pivot) {
                    break;
                }
            }
            loop {
                j -= 1;
                if !(nodes[j as usize].co[axis] > pivot && j > left) {
                    break;
                }
            }
        }
        nodes.swap(i as usize, right as usize);
        if i >= median {
            right = i - 1;
        }
        if i <= median {
            left = i + 1;
        }
    }
}
