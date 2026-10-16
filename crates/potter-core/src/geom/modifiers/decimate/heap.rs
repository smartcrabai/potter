//! Port of Blender's `BLI_heap` (`blenlib/intern/heap.cc`): a binary min-heap whose tie order depends
//! on its exact sift rules, which Decimate's collapse order relies on. Nodes are addressed by
//! [`NodeId`]; a node keeps its `ptr` (an element index for the caller) until it is popped or removed.

/// Handle of a live heap node. Stale handles are the caller's bug, as with Blender's `HeapNode *`.
pub(super) type NodeId = usize;

#[derive(Clone, Copy, Debug)]
struct Node {
    value: f32,
    index: usize,
    ptr: usize,
}

#[derive(Debug, Default)]
pub(super) struct Heap {
    tree: Vec<NodeId>,
    nodes: Vec<Node>,
    free: Vec<NodeId>,
}

impl Heap {
    pub(super) fn with_capacity(reserve: usize) -> Self {
        Self {
            tree: Vec::with_capacity(reserve),
            nodes: Vec::with_capacity(reserve),
            free: Vec::new(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// Value of the minimum node. The heap must not be empty.
    pub(super) fn top_value(&self) -> f32 {
        self.nodes[self.tree[0]].value
    }

    pub(super) fn insert(&mut self, value: f32, ptr: usize) -> NodeId {
        let id = if let Some(id) = self.free.pop() {
            self.nodes[id] = Node {
                value,
                index: self.tree.len(),
                ptr,
            };
            id
        } else {
            self.nodes.push(Node {
                value,
                index: self.tree.len(),
                ptr,
            });
            self.nodes.len() - 1
        };
        self.tree.push(id);
        let index = self.tree.len() - 1;
        self.up(index);
        id
    }

    /// Blender's `BLI_heap_insert_or_update`: inserts into an empty slot, otherwise updates the node.
    pub(super) fn insert_or_update(&mut self, slot: &mut Option<NodeId>, value: f32, ptr: usize) {
        match *slot {
            None => *slot = Some(self.insert(value, ptr)),
            Some(node) => self.update(node, value, ptr),
        }
    }

    /// Removes and returns the element pointer of the minimum node.
    pub(super) fn pop_min(&mut self) -> Option<usize> {
        let &top = self.tree.first()?;
        let ptr = self.nodes[top].ptr;
        self.free.push(top);
        let last = self.tree.len() - 1;
        if last > 0 {
            self.swap(0, last);
        }
        self.tree.pop();
        if !self.tree.is_empty() {
            self.down(0);
        }
        Some(ptr)
    }

    /// Removes a node by moving it to the root and popping it.
    pub(super) fn remove(&mut self, node: NodeId) {
        let mut index = self.nodes[node].index;
        while index > 0 {
            let parent = (index - 1) >> 1;
            self.swap(parent, index);
            index = parent;
        }
        self.pop_min();
    }

    /// Blender's `BLI_heap_node_value_update_ptr`: sets the pointer and re-sifts only on change.
    pub(super) fn update(&mut self, node: NodeId, value: f32, ptr: usize) {
        self.nodes[node].ptr = ptr;
        if value < self.nodes[node].value {
            self.nodes[node].value = value;
            let index = self.nodes[node].index;
            self.up(index);
        } else if value > self.nodes[node].value {
            self.nodes[node].value = value;
            let index = self.nodes[node].index;
            self.down(index);
        }
    }

    fn swap(&mut self, first: usize, second: usize) {
        self.tree.swap(first, second);
        let (a, b) = (self.tree[first], self.tree[second]);
        self.nodes[a].index = first;
        self.nodes[b].index = second;
    }

    fn less(&self, a: usize, b: usize) -> bool {
        self.nodes[self.tree[a]].value < self.nodes[self.tree[b]].value
    }

    fn up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) >> 1;
            if self.less(parent, index) {
                break;
            }
            self.swap(parent, index);
            index = parent;
        }
    }

    fn down(&mut self, mut index: usize) {
        let size = self.tree.len();
        loop {
            let left = (index << 1) + 1;
            let right = left + 1;
            let mut smallest = index;
            if left < size && self.less(left, smallest) {
                smallest = left;
            }
            if right < size && self.less(right, smallest) {
                smallest = right;
            }
            if smallest == index {
                break;
            }
            self.swap(index, smallest);
            index = smallest;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Heap;

    #[test]
    fn pops_in_value_order_and_removes_nodes() {
        let mut heap = Heap::with_capacity(4);
        let mut slots = [None; 5];
        for (ptr, value) in [3.0_f32, 1.0, 4.0, 1.0, 5.0].into_iter().enumerate() {
            heap.insert_or_update(&mut slots[ptr], value, ptr);
        }
        if let Some(node) = slots[2].take() {
            heap.remove(node);
        }
        let mut order = Vec::new();
        while let Some(ptr) = heap.pop_min() {
            order.push(ptr);
        }
        assert_eq!(order, vec![1, 3, 0, 4]);
    }
}
