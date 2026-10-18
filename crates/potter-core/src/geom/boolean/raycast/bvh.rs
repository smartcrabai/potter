//! Bounding-volume hierarchy over triangle boxes, culling the candidates of the float-ray tests in
//! the non-PWN classification (Blender `raycast_tree`: one `BVHTree` per boolean).
//!
//! Construction is a median split on the widest centroid axis, down to leaves of at most
//! [`LEAF_SIZE`] entries. A ray query returns the entries of every leaf whose node box the ray
//! crosses, so it contains every entry whose own box the ray crosses.

use glam::DVec3;

/// Largest number of entries in one leaf.
const LEAF_SIZE: usize = 4;

#[derive(Clone, Copy, Debug)]
struct Node {
    min: DVec3,
    max: DVec3,
    /// Leaf: first position in [`TriBvh::entries`]. Inner node: index of the right child (the left
    /// child is the next node).
    offset: usize,
    /// Leaf: number of entries, at least one. Inner node: zero.
    count: usize,
}

/// Box hierarchy over the padded boxes of indexed entries.
#[derive(Debug, Default)]
pub(super) struct TriBvh {
    nodes: Vec<Node>,
    entries: Vec<usize>,
}

impl TriBvh {
    /// Builds the hierarchy over `bounds[e]`, each box grown by `pad` on every side.
    ///
    /// Cost: O(T log T) time and O(T) memory for T entries.
    pub(super) fn new(bounds: &[(DVec3, DVec3)], pad: f64) -> Self {
        let padded: Vec<(DVec3, DVec3)> = bounds
            .iter()
            .map(|&(min, max)| (min - DVec3::splat(pad), max + DVec3::splat(pad)))
            .collect();
        let centers: Vec<DVec3> = padded.iter().map(|&(min, max)| (min + max) * 0.5).collect();
        let mut entries: Vec<usize> = (0..padded.len()).collect();
        let mut bvh = Self {
            nodes: Vec::with_capacity(2 * padded.len() / LEAF_SIZE + 1),
            entries: Vec::new(),
        };
        if !entries.is_empty() {
            bvh.build(&mut entries, 0, &padded, &centers);
        }
        bvh.entries = entries;
        bvh
    }

    /// Appends the subtree over `entries`, which occupy positions `start..` of the final array.
    ///
    /// Each level partitions its slice with `select_nth_unstable_by`, so a level costs O(T) and
    /// the whole build O(T log T).
    fn build(
        &mut self,
        entries: &mut [usize],
        start: usize,
        boxes: &[(DVec3, DVec3)],
        centers: &[DVec3],
    ) {
        let (min, max) = entries.iter().fold(
            (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)),
            |(lo, hi), &e| (lo.min(boxes[e].0), hi.max(boxes[e].1)),
        );
        if entries.len() <= LEAF_SIZE {
            self.nodes.push(Node {
                min,
                max,
                offset: start,
                count: entries.len(),
            });
            return;
        }
        let (center_min, center_max) = entries.iter().fold(
            (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)),
            |(lo, hi), &e| (lo.min(centers[e]), hi.max(centers[e])),
        );
        let extent = center_max - center_min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        let mid = entries.len() / 2;
        entries.select_nth_unstable_by(mid, |&a, &b| centers[a][axis].total_cmp(&centers[b][axis]));
        let index = self.nodes.len();
        self.nodes.push(Node {
            min,
            max,
            offset: 0,
            count: 0,
        });
        let (left, right) = entries.split_at_mut(mid);
        self.build(left, start, boxes, centers);
        let right_index = self.nodes.len();
        self.build(right, start + mid, boxes, centers);
        self.nodes[index].offset = right_index;
    }

    /// Appends to `out` every entry whose padded box the ray `origin + s * direction`, `s >= 0`,
    /// crosses, and possibly other entries of the same leaves. `stack` is scratch space.
    ///
    /// Every component of `direction` must be nonzero. Cost: O(log T + k) for k returned entries
    /// and the nodes they lie under.
    pub(super) fn ray_candidates(
        &self,
        origin: DVec3,
        direction: DVec3,
        stack: &mut Vec<usize>,
        out: &mut Vec<usize>,
    ) {
        stack.clear();
        if self.nodes.is_empty() {
            return;
        }
        let inverse = DVec3::ONE / direction;
        stack.push(0);
        while let Some(index) = stack.pop() {
            let node = self.nodes[index];
            if !ray_crosses_box(node.min, node.max, origin, inverse) {
                continue;
            }
            if node.count > 0 {
                out.extend_from_slice(&self.entries[node.offset..node.offset + node.count]);
            } else {
                stack.push(node.offset);
                stack.push(index + 1);
            }
        }
    }
}

/// Slab test of the ray `origin + s * direction`, `s >= 0`, against a box. `inverse` is the
/// componentwise reciprocal of the direction.
fn ray_crosses_box(min: DVec3, max: DVec3, origin: DVec3, inverse: DVec3) -> bool {
    let near = (min - origin) * inverse;
    let far = (max - origin) * inverse;
    let enter = near.min(far).max_element().max(0.0);
    let exit = near.max(far).min_element();
    enter <= exit
}

#[cfg(test)]
mod tests {
    use super::TriBvh;
    use glam::DVec3;

    #[test]
    fn ray_candidates_contain_every_entry_the_ray_crosses() {
        // Entry `i * 10 + j` is the unit box at (2i, 2j, 0).
        let boxes: Vec<(DVec3, DVec3)> = (0..10)
            .flat_map(|i| {
                (0..10).map(move |j| {
                    let min = DVec3::new(2.0 * f64::from(i), 2.0 * f64::from(j), 0.0);
                    (min, min + DVec3::ONE)
                })
            })
            .collect();
        let bvh = TriBvh::new(&boxes, 0.0);
        let (mut stack, mut out) = (Vec::new(), Vec::new());
        // Runs along x at y = 4.5, z = 0.5: crosses exactly the boxes with j = 2.
        bvh.ray_candidates(
            DVec3::new(-1.0, 4.5, 0.5),
            DVec3::new(1.0, 1.0e-9, 1.0e-9),
            &mut stack,
            &mut out,
        );
        for entry in (0..boxes.len()).filter(|e| e % 10 == 2) {
            assert!(out.contains(&entry), "missing candidate {entry}");
        }
    }

    #[test]
    fn empty_hierarchy_has_no_candidates() {
        let bvh = TriBvh::new(&[], 0.0);
        let (mut stack, mut out) = (Vec::new(), Vec::new());
        bvh.ray_candidates(DVec3::ZERO, DVec3::new(1.0, 1.0, 1.0), &mut stack, &mut out);
        assert_eq!(out, Vec::<usize>::new());
    }
}
