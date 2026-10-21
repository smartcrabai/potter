// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenlib/intern/BLI_heap_simple.cc, SPDX-FileCopyrightText: 2013 Blender Authors,
//   GPL-2.0-or-later.

/// Blender's `BLI_heapsimple` (`BLI_heap_simple.cc`): a binary min-heap. Insertion sifts up while the
/// parent is larger; removal takes the root and sifts the last node down from the root. Equal keys
/// therefore pop in the order these sift rules give, as in Blender.
pub(crate) struct HeapSimple<T: Copy> {
    tree: Vec<(f32, T)>,
}

impl<T: Copy> HeapSimple<T> {
    /// `BLI_heapsimple_new`.
    pub(crate) fn new() -> Self {
        Self { tree: Vec::new() }
    }

    /// `BLI_heapsimple_insert` (`heapsimple_up`).
    pub(crate) fn insert(&mut self, value: f32, payload: T) {
        let mut index = self.tree.len();
        self.tree.push((value, payload));
        while index > 0 {
            let parent = (index - 1) >> 1;
            if value >= self.tree[parent].0 {
                break;
            }
            self.tree[index] = self.tree[parent];
            index = parent;
        }
        self.tree[index] = (value, payload);
    }

    /// `BLI_heapsimple_pop_min`: the root is returned and the last node is sifted down from the root.
    pub(crate) fn pop_min(&mut self) -> Option<T> {
        let (_, payload) = *self.tree.first()?;
        let last = self.tree.pop()?;
        if !self.tree.is_empty() {
            self.sift_down(last);
        }
        Some(payload)
    }

    /// `heapsimple_down` from the root, placing `init` (`BLI_heap_simple.cc`).
    fn sift_down(&mut self, init: (f32, T)) {
        let size = self.tree.len();
        let active_value = init.0;
        let mut index = 0;
        self.tree[index].0 = active_value;
        loop {
            let left = 2 * index + 1;
            let right = left + 1;
            let mut smallest = index;
            if left < size && self.tree[left].0 < active_value {
                smallest = left;
            }
            if right < size && self.tree[right].0 < self.tree[smallest].0 {
                smallest = right;
            }
            if smallest == index {
                break;
            }
            self.tree[index] = self.tree[smallest];
            index = smallest;
            self.tree[index].0 = active_value;
        }
        self.tree[index].1 = init.1;
    }
}
