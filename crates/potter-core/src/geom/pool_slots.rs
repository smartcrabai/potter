// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenlib/intern/BLI_mempool.cc, SPDX-FileCopyrightText: 2008 Blender Authors, GPL-2.0-or-later.

/// Blender's `BLI_mempool` slot order for one element type. `BLI_mempool_alloc` takes the head of the
/// free list, the most recently freed slot, and then never-used slots in order. Blender emits elements
/// in pool order (`BM_mesh_bm_to_me`), so an element's slot is its output position.
#[derive(Default)]
pub(crate) struct PoolSlots {
    fresh: usize,
    freed: Vec<usize>,
}

impl PoolSlots {
    pub(crate) fn alloc(&mut self) -> usize {
        if let Some(slot) = self.freed.pop() {
            return slot;
        }
        self.fresh += 1;
        self.fresh - 1
    }

    pub(crate) fn release(&mut self, slot: usize) {
        self.freed.push(slot);
    }
}
