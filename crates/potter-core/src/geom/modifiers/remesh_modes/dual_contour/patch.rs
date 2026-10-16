// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   intern/dualcon/intern/octree.cpp (trace, findPaths, combinePaths, combineSinglePath, deletePath,
//   patch, patchSplit, patchSplitSingle, connectFace, patchAdjacent, locateCell, getFacePoint,
//   findPair, compressRing), SPDX-FileCopyrightText: 2011-2023 Blender Authors, GPL-2.0-or-later.

//! Detection and patching of open dual-surface rings (Blender's `trace` family).
//!
//! Blender threads `PathList *&` and `PathElement *&` through these routines; each reference is
//! expressed here as an `Option<u32>` return value or a sentinel list node whose `next` field plays
//! the role of the Blender variable. Node pointers become arena indices; the order in which rings
//! are prepended, joined, reversed, and split is preserved exactly, since it fixes the output order.

use super::{
    invariant,
    octree::{Arena, Child, Octree, PathElement, PathList, child_slots, offset_position},
    tables::{CELL_PROC_FACE_MASK, EDGEMAP, FACE_MAP, FACE_PROC_FACE_MASK, VERTMAP},
};
use crate::error::Result;

impl Octree {
    fn push_element(&mut self, pos: [i32; 3], next: Option<u32>) -> Result<u32> {
        self.reserve_work_item(Arena::Element)?;
        self.elements.push(PathElement { pos, next });
        Ok((self.elements.len() - 1) as u32)
    }

    fn push_list(&mut self, list: PathList) -> Result<u32> {
        self.reserve_work_item(Arena::List)?;
        self.lists.push(list);
        Ok((self.lists.len() - 1) as u32)
    }

    /// Sentinel list node whose `next` is the Blender head variable.
    fn sentinel(&mut self, next: Option<u32>) -> Result<u32> {
        self.push_list(PathList {
            head: None,
            tail: None,
            length: 0,
            next,
        })
    }

    fn element_equal(&self, first: u32, second: u32) -> bool {
        self.elements[first as usize].pos == self.elements[second as usize].pos
    }

    /// `trace()`: patches every closed hole in the octree, top-down from the root.
    pub(super) fn trace_root(&mut self) -> Result<()> {
        let root = Self::root();
        let (dimen, depth) = (self.dimen, self.max_depth);
        self.trace_node(root, [0; 3], dimen, depth)?;
        Ok(())
    }

    /// `trace(Node*, int*, int, int, PathList*&)`: returns the incomplete paths of child 0.
    fn trace_node(&mut self, node: u32, st: [i32; 3], len: i32, depth: i32) -> Result<Option<u32>> {
        let len = len >> 1;
        let children = self.internals[node as usize].children;
        let mut child_paths = [None::<u32>; 8];
        let mut nst = [[0_i32; 3]; 8];
        for octant in 0..8 {
            nst[octant] = offset_position(st, VERTMAP[octant], len);
        }
        for octant in 0..8 {
            if let Child::Internal(child) = children[octant] {
                child_paths[octant] = self.trace_node(child, nst[octant], len, depth - 1)?;
            }
        }

        let mut connectors = [None::<u32>; 12];
        for (index, mask) in CELL_PROC_FACE_MASK.iter().enumerate() {
            let nodes = [children[mask[0]], children[mask[1]]];
            let positions = [nst[mask[0]], nst[mask[1]]];
            self.find_paths(
                nodes,
                [depth - 1, depth - 1],
                positions,
                depth - 1,
                mask[2],
                &mut connectors[index],
            )?;
        }

        let mut rings = None::<u32>;
        let [mut p0, p1, mut p2, p3, mut p4, p5, mut p6, p7] = child_paths;
        self.combine_paths(&mut p0, p1, connectors[8], &mut rings)?;
        self.combine_paths(&mut p2, p3, connectors[9], &mut rings)?;
        self.combine_paths(&mut p4, p5, connectors[10], &mut rings)?;
        self.combine_paths(&mut p6, p7, connectors[11], &mut rings)?;

        self.combine_paths(&mut p0, p2, connectors[4], &mut rings)?;
        self.combine_paths(&mut p4, p6, connectors[5], &mut rings)?;
        self.combine_paths(&mut p0, None, connectors[6], &mut rings)?;
        self.combine_paths(&mut p4, None, connectors[7], &mut rings)?;

        self.combine_paths(&mut p0, p4, connectors[0], &mut rings)?;
        self.combine_paths(&mut p0, None, connectors[1], &mut rings)?;
        self.combine_paths(&mut p0, None, connectors[2], &mut rings)?;
        self.combine_paths(&mut p0, None, connectors[3], &mut rings)?;

        if rings.is_some() {
            self.patch(node, st, len << 1, rings)?;
        }
        Ok(p0)
    }

    /// `findPaths`: collects the open connector segments on one face shared by two nodes.
    fn find_paths(
        &mut self,
        nodes: [Child; 2],
        depths: [i32; 2],
        st: [[i32; 3]; 2],
        maxdep: i32,
        direction: usize,
        paths: &mut Option<u32>,
    ) -> Result<()> {
        if nodes.contains(&Child::Empty) {
            return Ok(());
        }
        let leaf = nodes.map(|node| matches!(node, Child::Leaf(_)));
        if leaf[0] && leaf[1] {
            let index = usize::from(depths[0] != maxdep);
            let face = 2 * direction + (1 - index);
            let Child::Leaf(leaf_id) = nodes[index] else {
                return Err(invariant("dual contour face test expected a leaf"));
            };
            if self.face_parity(leaf_id, face) {
                let second = self.push_element(st[1], None)?;
                let first = self.push_element(st[0], Some(second))?;
                let list = self.push_list(PathList {
                    head: Some(first),
                    tail: Some(second),
                    length: 2,
                    next: *paths,
                })?;
                *paths = Some(list);
            }
            return Ok(());
        }

        let children = nodes.map(|node| child_slots(&self.internals, node));
        let mut positions = [[[0_i32; 3]; 8]; 2];
        for side in 0..2 {
            if !leaf[side] {
                let len = self.dimen >> (self.max_depth - depths[side] + 1);
                for octant in 0..8 {
                    positions[side][octant] = offset_position(st[side], VERTMAP[octant], len);
                }
            }
        }
        for face in FACE_PROC_FACE_MASK[direction] {
            let next: [Child; 2] = std::array::from_fn(|side| {
                if leaf[side] {
                    nodes[side]
                } else {
                    children[side][face[side]]
                }
            });
            let next_depths: [i32; 2] = std::array::from_fn(|side| {
                if leaf[side] {
                    depths[side]
                } else {
                    depths[side] - 1
                }
            });
            let next_positions: [[i32; 3]; 2] = std::array::from_fn(|side| {
                if leaf[side] {
                    st[side]
                } else {
                    positions[side][face[side]]
                }
            });
            self.find_paths(
                next,
                next_depths,
                next_positions,
                maxdep - 1,
                face[2],
                paths,
            )?;
        }
        Ok(())
    }

    /// `combinePaths`: hooks the connector segments into `list1` and `list2`, moves closed loops
    /// into `rings`, and appends the leftovers to `list1`.
    fn combine_paths(
        &mut self,
        list1: &mut Option<u32>,
        list2: Option<u32>,
        paths: Option<u32>,
        rings: &mut Option<u32>,
    ) -> Result<()> {
        let head1 = self.sentinel(*list1)?;
        let head2 = self.sentinel(list2)?;
        let new_list = self.sentinel(None)?;
        let ring_list = self.sentinel(*rings)?;
        let single = self.sentinel(None)?;

        let mut remaining = paths;
        while let Some(entry) = remaining {
            remaining = self.lists[entry as usize].next;
            self.lists[entry as usize].next = None;
            let mut singlist = entry;

            for head in [head1, head2, new_list] {
                let mut previous: Option<u32> = None;
                let mut current = self.lists[head as usize].next;
                while let Some(candidate) = current {
                    let pre = previous.unwrap_or(head);
                    self.lists[single as usize].next = Some(singlist);
                    if let Some(joined) =
                        self.combine_single_path(pre, candidate, single, singlist)?
                    {
                        singlist = joined;
                        current = self.lists[pre as usize].next;
                    } else {
                        previous = Some(candidate);
                        current = self.lists[candidate as usize].next;
                    }
                }
            }

            let head = self.lists[singlist as usize].head;
            let tail = self.lists[singlist as usize].tail;
            if let (Some(head), Some(tail)) = (head, tail) {
                if self.element_equal(head, tail) {
                    let new_head = self.elements[head as usize].next;
                    self.lists[singlist as usize].head = new_head;
                    self.lists[singlist as usize].length -= 1;
                    if let Some(new_head) = new_head {
                        self.elements[tail as usize].next = Some(new_head);
                    }
                    self.lists[singlist as usize].next = self.lists[ring_list as usize].next;
                    self.lists[ring_list as usize].next = Some(singlist);
                } else {
                    self.lists[singlist as usize].next = self.lists[new_list as usize].next;
                    self.lists[new_list as usize].next = Some(singlist);
                }
            }
        }

        // Append list2 and the new paths to the end of list1.
        let first1 = self.lists[head1 as usize].next;
        let second = self.lists[head2 as usize].next;
        let fresh = self.lists[new_list as usize].next;
        let tail_of_list1 = if let Some(first) = first1 {
            let mut last = first;
            while let Some(next) = self.lists[last as usize].next {
                last = next;
            }
            self.lists[last as usize].next = second;
            Some(last)
        } else {
            self.lists[head1 as usize].next = second;
            second
        };
        match tail_of_list1 {
            Some(mut last) => {
                while let Some(next) = self.lists[last as usize].next {
                    last = next;
                }
                self.lists[last as usize].next = fresh;
            }
            None => {
                self.lists[head1 as usize].next = fresh;
            }
        }

        *list1 = self.lists[head1 as usize].next;
        *rings = self.lists[ring_list as usize].next;
        Ok(())
    }

    /// `combineSinglePath`: joins two paths if they share an endpoint. Returns the joined path.
    fn combine_single_path(
        &mut self,
        pre1: u32,
        list1: u32,
        pre2: u32,
        list2: u32,
    ) -> Result<Option<u32>> {
        let (Some(head1), Some(tail1), Some(head2), Some(tail2)) = (
            self.lists[list1 as usize].head,
            self.lists[list1 as usize].tail,
            self.lists[list2 as usize].head,
            self.lists[list2 as usize].tail,
        ) else {
            return Ok(None);
        };
        if self.element_equal(head1, head2) || self.element_equal(tail1, tail2) {
            if self.lists[list1 as usize].length < self.lists[list2 as usize].length {
                self.reverse_path(list1);
            } else {
                self.reverse_path(list2);
            }
        }

        let (Some(head1), Some(tail1), Some(head2), Some(tail2)) = (
            self.lists[list1 as usize].head,
            self.lists[list1 as usize].tail,
            self.lists[list2 as usize].head,
            self.lists[list2 as usize].tail,
        ) else {
            return Ok(None);
        };
        let length = self.lists[list1 as usize].length + self.lists[list2 as usize].length - 1;

        if self.element_equal(head1, tail2) {
            let temp = self.elements[head1 as usize].next;
            self.elements[tail2 as usize].next = temp;
            let joined = self.push_list(PathList {
                head: Some(head2),
                tail: Some(tail1),
                length,
                next: None,
            })?;
            self.delete_path(pre1, list1);
            self.delete_path(pre2, list2);
            return Ok(Some(joined));
        }
        if self.element_equal(tail1, head2) {
            let temp = self.elements[head2 as usize].next;
            self.elements[tail1 as usize].next = temp;
            let joined = self.push_list(PathList {
                head: Some(head1),
                tail: Some(tail2),
                length,
                next: None,
            })?;
            self.delete_path(pre1, list1);
            self.delete_path(pre2, list2);
            return Ok(Some(joined));
        }
        Ok(None)
    }

    /// Reverses a path in place: swaps head and tail and flips element links.
    fn reverse_path(&mut self, list: u32) {
        let Some(head) = self.lists[list as usize].head else {
            return;
        };
        let mut previous = head;
        let mut next = self.elements[head as usize].next;
        self.elements[head as usize].next = None;
        while let Some(current) = next {
            let following = self.elements[current as usize].next;
            self.elements[current as usize].next = Some(previous);
            previous = current;
            next = following;
        }
        self.lists[list as usize].tail = self.lists[list as usize].head;
        self.lists[list as usize].head = Some(previous);
    }

    /// `deletePath`: unlinks `current` from the list whose predecessor link is `pre`.
    fn delete_path(&mut self, pre: u32, current: u32) {
        let next = self.lists[current as usize].next;
        self.lists[pre as usize].next = next;
    }

    /// `patch(Node*, int*, int, PathList*)`: closes rings inside one node.
    fn patch(&mut self, node: u32, st: [i32; 3], len: i32, rings: Option<u32>) -> Result<()> {
        if len == self.mindimen {
            return Err(invariant(
                "dual contour ring reached the minimal cell size without closing",
            ));
        }
        let (x0, x1) = self.patch_split(node, st, len, rings, 0)?;
        let (y0, y1) = self.patch_split(node, st, len, x0, 1)?;
        let (y2, y3) = self.patch_split(node, st, len, x1, 1)?;
        let (z0, z1) = self.patch_split(node, st, len, y0, 2)?;
        let (z2, z3) = self.patch_split(node, st, len, y1, 2)?;
        let (z4, z5) = self.patch_split(node, st, len, y2, 2)?;
        let (z6, z7) = self.patch_split(node, st, len, y3, 2)?;
        let zlists = [z0, z1, z2, z3, z4, z5, z6, z7];

        let child_len = len >> 1;
        for (octant, list) in zlists.iter().enumerate() {
            let Some(list) = *list else {
                continue;
            };
            let child_origin = offset_position(st, VERTMAP[octant], child_len);
            match self.internals[node as usize].children[octant] {
                Child::Internal(child) => {
                    self.patch(child, child_origin, child_len, Some(list))?;
                }
                _ => return Err(invariant("dual contour ring child is not an internal node")),
            }
        }
        Ok(())
    }

    /// `patchSplit`: distributes rings to the two halves of `node` along `direction`.
    fn patch_split(
        &mut self,
        node: u32,
        st: [i32; 3],
        len: i32,
        rings: Option<u32>,
        direction: usize,
    ) -> Result<(Option<u32>, Option<u32>)> {
        let mut first = None;
        let mut second = None;
        let mut remaining = rings;
        while let Some(list) = remaining {
            let head = self.lists[list as usize].head;
            self.patch_split_single(node, st, len, head, direction, &mut first, &mut second)?;
            remaining = self.lists[list as usize].next;
        }
        Ok((first, second))
    }

    /// `patchSplitSingle`: splits one ring at its crossings with the midplane, connects the two
    /// halves across the face, and recurses into the resulting rings.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors Blender's patchSplitSingle signature"
    )]
    fn patch_split_single(
        &mut self,
        node: u32,
        st: [i32; 3],
        len: i32,
        head: Option<u32>,
        direction: usize,
        first: &mut Option<u32>,
        second: &mut Option<u32>,
    ) -> Result<()> {
        let Some(head) = head else {
            return Ok(());
        };
        let position = st[direction] + len / 2;
        let (side, pre1, pre2) = self.find_pair(head, position, direction);
        if side != 0 {
            let target = if side == -1 { first } else { second };
            let ring = self.push_list(PathList {
                head: Some(head),
                tail: None,
                length: 0,
                next: *target,
            })?;
            *target = Some(ring);
            return Ok(());
        }
        let (Some(pre1), Some(pre2)) = (pre1, pre2) else {
            return Err(invariant("dual contour ring split found no crossing pair"));
        };

        let next1 = self.elements[pre1 as usize].next;
        let next2 = self.elements[pre2 as usize].next;
        self.elements[pre1 as usize].next = next2;
        self.elements[pre2 as usize].next = next1;

        self.connect_face(node, st, len, direction, pre1, pre2)?;

        let mut ring1 = Some(pre1);
        let mut ring2 = Some(pre2);
        if let Some(ring) = ring1 {
            let next = self.elements[ring as usize].next.unwrap_or(ring);
            if self.element_equal(ring, next) {
                if ring == next {
                    ring1 = None;
                } else {
                    let temp = next;
                    self.elements[ring as usize].next = self.elements[temp as usize].next;
                }
            }
        }
        if let Some(ring) = ring2 {
            let next = self.elements[ring as usize].next.unwrap_or(ring);
            if self.element_equal(ring, next) {
                if ring == next {
                    ring2 = None;
                } else {
                    let temp = next;
                    self.elements[ring as usize].next = self.elements[temp as usize].next;
                }
            }
        }

        let ring1 = self.compress_ring(ring1);
        let ring2 = self.compress_ring(ring2);
        self.patch_split_single(node, st, len, ring1, direction, first, second)?;
        self.patch_split_single(node, st, len, ring2, direction, first, second)
    }

    /// `connectFace`: walks the face between two rings' halves, creating the cells and walk
    /// elements that stitch the patch together.
    fn connect_face(
        &mut self,
        node: u32,
        st: [i32; 3],
        len: i32,
        direction: usize,
        f1: u32,
        f2: u32,
    ) -> Result<()> {
        let position = st[direction] + len / 2;
        let xdir = (direction + 1) % 3;
        let ydir = (direction + 2) % 3;

        let f2_next = self.elements[f2 as usize]
            .next
            .ok_or_else(|| invariant("dual contour ring element has no successor"))?;
        let (x1, y1, p1, q1) = self.get_face_point(f2_next, direction);
        let (x2, y2, p2, q2) = self.get_face_point(f2, direction);

        let dx = x2 as f32 + p2 - x1 as f32 - p1;
        let dy = y2 as f32 + q2 - y1 as f32 - q1;

        let mut rx = p1;
        let mut ry = q1;
        let mut incx: i32 = 1;
        let mut incy: i32 = 1;
        let (mut lx, mut ly) = (x1, y1);
        let (mut hx, mut hy) = (x2, y2);
        if x2 < x1 {
            incx = -1;
            rx = 1.0 - rx;
            lx = x2;
            hx = x1;
        }
        if y2 < y1 {
            incy = -1;
            ry = 1.0 - ry;
            ly = y2;
            hy = y1;
        }

        let sx = dx * incx as f32;
        let sy = dy * incy as f32;

        let mut ori = [0_i32; 3];
        ori[direction] = position / self.mindimen;
        ori[xdir] = x1;
        ori[ydir] = y1;

        let mut cur_n = f1;
        let mut cur_p = f2_next;
        let mut leaf_cur_n = self.locate_leaf_below(node, len, self.elements[f1 as usize].pos)?;
        let mut leaf_cur_p =
            self.locate_leaf_below(node, len, self.elements[f2_next as usize].pos)?;

        while ori[xdir] != x2 || ori[ydir] != y2 {
            let mut choice: u8;
            let mut next: i32;
            if sy * (1.0 - rx) > sx * (1.0 - ry) {
                choice = 1;
                next = ori[ydir] + incy;
                if next < ly || next > hy {
                    choice = 4;
                    next = ori[xdir] + incx;
                }
            } else {
                choice = 2;
                next = ori[xdir] + incx;
                if next < lx || next > hx {
                    choice = 3;
                    next = ori[ydir] + incy;
                }
            }

            let walkdir: usize;
            let inc: i32;
            let alpha: f32;
            if choice & 1 != 0 {
                ori[ydir] = next;
                if choice == 1 {
                    rx += if sy == 0.0 { 0.0 } else { (1.0 - ry) * sx / sy };
                    ry = 0.0;
                }
                walkdir = 2;
                inc = incy;
                alpha = if x2 < x1 { 1.0 - rx } else { rx };
            } else {
                ori[xdir] = next;
                if choice == 2 {
                    ry += if sx == 0.0 { 0.0 } else { (1.0 - rx) * sy / sx };
                    rx = 0.0;
                }
                walkdir = 1;
                inc = incx;
                alpha = if y2 < y1 { 1.0 - ry } else { ry };
            }

            let nori = ori.map(|coordinate| coordinate * self.mindimen);
            let (leaf_n, st_n) = self.locate_cell(node, st, len, nori, direction, 1)?;
            let (leaf_p, st_p) = self.locate_cell(node, st, len, nori, direction, 0)?;

            if self.elements[cur_n as usize].pos != st_n {
                let after = self.elements[cur_n as usize]
                    .next
                    .ok_or_else(|| invariant("dual contour ring element has no successor"))?;
                let new_n = if self.elements[after as usize].pos == st_n {
                    after
                } else {
                    let element = self.push_element(st_n, Some(after))?;
                    self.elements[cur_n as usize].next = Some(element);
                    element
                };
                leaf_cur_n = self.patch_adjacent(
                    node,
                    len,
                    self.elements[cur_n as usize].pos,
                    leaf_cur_n,
                    self.elements[new_n as usize].pos,
                    leaf_n,
                    walkdir,
                    inc,
                    direction,
                    1,
                    alpha,
                );
                cur_n = new_n;
            }

            if self.elements[cur_p as usize].pos != st_p {
                let new_p = if self.elements[f2 as usize].pos == st_p {
                    f2
                } else {
                    let element = self.push_element(st_p, Some(cur_p))?;
                    self.elements[f2 as usize].next = Some(element);
                    element
                };
                leaf_cur_p = self.patch_adjacent(
                    node,
                    len,
                    self.elements[cur_p as usize].pos,
                    leaf_cur_p,
                    self.elements[new_p as usize].pos,
                    leaf_p,
                    walkdir,
                    inc,
                    direction,
                    0,
                    alpha,
                );
                cur_p = new_p;
            }
        }
        Ok(())
    }

    /// `patchAdjacent`: flips the two edges that a patch step crosses and returns the updated
    /// second leaf.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors Blender's patchAdjacent signature"
    )]
    fn patch_adjacent(
        &mut self,
        _node: u32,
        _len: i32,
        _st1: [i32; 3],
        leaf1: u32,
        _st2: [i32; 3],
        leaf2: u32,
        walkdir: usize,
        inc: i32,
        direction: usize,
        side: i32,
        alpha: f32,
    ) -> u32 {
        let (dir, walk) = (direction as i32, walkdir as i32);
        let edgedir = (dir + (3 - walk)) % 3;
        let incdir = (dir + walk) % 3;
        let ind1 = if edgedir == 1 {
            (dir + 3 - edgedir) % 3 - 1
        } else {
            2 - (dir + 3 - edgedir) % 3
        };
        let ind2 = if edgedir == 1 {
            (incdir + 3 - edgedir) % 3 - 1
        } else {
            2 - (incdir + 3 - edgedir) % 3
        };
        let forward = i32::from(inc > 0);
        let eind1 = (edgedir << 2) | (side << ind1) | ((forward) << ind2);
        let eind2 = (edgedir << 2) | (side << ind1) | ((1 - forward) << ind2);

        self.flip_edge_with_offset(leaf1, eind1 as usize, alpha);
        self.flip_edge_with_offset(leaf2, eind2 as usize, alpha);
        leaf2
    }

    /// `locateCell`: finds (creating if needed) the minimal cell on one side of the midplane at
    /// `ori`. Returns the leaf and its origin.
    fn locate_cell(
        &mut self,
        node: u32,
        st: [i32; 3],
        len: i32,
        ori: [i32; 3],
        direction: usize,
        side: i32,
    ) -> Result<(u32, [i32; 3])> {
        let len = len >> 1;
        let mut index = 0_usize;
        for axis in 0..3 {
            index <<= 1;
            let high = if axis == direction && side == 1 {
                ori[axis] > st[axis] + len
            } else {
                ori[axis] >= st[axis] + len
            };
            if high {
                index |= 1;
            }
        }
        let rst = offset_position(st, VERTMAP[index], len);
        match self.internals[node as usize].children[index] {
            Child::Leaf(leaf) => Ok((leaf, rst)),
            Child::Internal(child) => self.locate_cell(child, rst, len, ori, direction, side),
            Child::Empty => {
                if len == self.mindimen {
                    let leaf = self.new_leaf()?;
                    self.internals[node as usize].children[index] = Child::Leaf(leaf);
                    Ok((leaf, rst))
                } else {
                    let child = self.new_internal()?;
                    self.internals[node as usize].children[index] = Child::Internal(child);
                    self.locate_cell(child, rst, len, ori, direction, side)
                }
            }
        }
    }

    /// `getFacePoint`: the average crossing point of a cell's face, as integer cell coordinates
    /// plus fractional offsets in the two in-plane directions.
    fn get_face_point(&self, element: u32, direction: usize) -> (i32, i32, f32, f32) {
        let position = self.elements[element as usize].pos;
        let mut sum = [0.0_f32; 3];
        let mut count = 0_u32;
        for edge_index in FACE_MAP[direction * 2] {
            let start = offset_position(position, VERTMAP[EDGEMAP[edge_index][0]], self.mindimen);
            if let Some(point) = self.edge_intersection_checked(start, edge_index / 4) {
                for axis in 0..3 {
                    sum[axis] += point[axis];
                }
                count += 1;
            }
        }
        let average = if count == 0 {
            position.map(|coordinate| coordinate as f32)
        } else {
            sum.map(|value| value / count as f32)
        };
        let xdir = (direction + 1) % 3;
        let ydir = (direction + 2) % 3;
        let x = position[xdir] >> self.minshift;
        let y = position[ydir] >> self.minshift;
        let p = (average[xdir] - position[xdir] as f32) / self.mindimen as f32;
        let q = (average[ydir] - position[ydir] as f32) / self.mindimen as f32;
        (x, y, p, q)
    }

    /// `findPair`: locates the pair of consecutive ring elements that straddle the midplane.
    /// Returns `(side, pre1, pre2)`; `side != 0` means the whole ring is on one side.
    fn find_pair(
        &self,
        head: u32,
        position: i32,
        direction: usize,
    ) -> (i32, Option<u32>, Option<u32>) {
        let side_of = |element: u32| -> i32 {
            if self.elements[element as usize].pos[direction] < position {
                -1
            } else {
                1
            }
        };
        let side = side_of(head);
        let anchor = head;
        let mut previous1 = head;
        let mut current = self.elements[head as usize].next.unwrap_or(head);
        while current != anchor && side_of(current) == side {
            previous1 = current;
            current = self.elements[current as usize].next.unwrap_or(anchor);
        }
        if current == anchor {
            return (side, None, None);
        }

        let side = side_of(current);
        let mut previous2 = current;
        current = self.elements[current as usize].next.unwrap_or(anchor);
        while side_of(current) == side {
            previous2 = current;
            current = self.elements[current as usize].next.unwrap_or(anchor);
        }

        if side == -1 {
            std::mem::swap(&mut previous1, &mut previous2);
        }
        (0, Some(previous1), Some(previous2))
    }

    /// `compressRing`: removes consecutive duplicate positions from a ring. Returns the anchor
    /// element, or `None` if the ring collapsed.
    fn compress_ring(&mut self, ring: Option<u32>) -> Option<u32> {
        let ring = ring?;
        let Some(first_next) = self.elements[ring as usize].next else {
            return Some(ring);
        };
        let mut current = self.elements[first_next as usize]
            .next
            .unwrap_or(first_next);
        let mut previous = first_next;
        let mut previous_previous = ring;
        let mut anchor = Some(previous_previous);
        loop {
            let mut collapsed = false;
            while self.element_equal(current, previous_previous) {
                if current == previous_previous {
                    anchor = None;
                    collapsed = true;
                    break;
                }
                self.elements[previous_previous as usize].next =
                    self.elements[current as usize].next;
                previous = self.elements[previous_previous as usize]
                    .next
                    .unwrap_or(previous_previous);
                current = self.elements[previous as usize].next.unwrap_or(previous);
                anchor = Some(previous_previous);
            }
            if collapsed || anchor.is_none() {
                break;
            }
            previous_previous = previous;
            previous = current;
            current = self.elements[current as usize].next.unwrap_or(current);
            if anchor == Some(previous_previous) {
                break;
            }
        }
        anchor
    }
}
