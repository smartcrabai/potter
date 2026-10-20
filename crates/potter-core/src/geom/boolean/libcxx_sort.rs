//! Sort order of Blender's `IMesh::populate_vert`, which decides the stored direction of tie edges.
//!
//! `populate_vert` fills `vert_` in first-seen order (`blenlib/intern/mesh_intersect.cc:559-577`)
//! and sorts it with `parallel_sort` (`blenlib/BLI_sort.hh:11-20`, `tbb::parallel_sort` in the TBB
//! build) using the comparator at `mesh_intersect.cc:583-594`: input vertices by ascending origin,
//! created vertices after them by id. Two input vertices with the same origin tie. On macOS the sort
//! is libc++ `std::sort` (LLVM pdqsort, `__algorithm/sort.h` in the macOS SDK), which is not stable,
//! so [`libcxx_sort_by`] reproduces that algorithm rather than a stable sort.
//!
//! Corpus limits: Blender 5.2.2 matched this routine on all 106 `populate_vert` calls of the EXACT
//! boolean corpus (up to 40 vertices), and it reproduces every oracle tie edge. The corpus never
//! reaches the heap fallback at depth zero (`__partial_sort`); the unit tests check that path against
//! libc++ `std::partial_sort` directly. The fallback treats the element as non-arithmetic, as
//! Blender's `const Vert *` is. libstdc++ (Linux) orders ties differently in intermediate calls but
//! gave the same oracle tie edges, so the port models macOS libc++ only.

/// Sorts `v` as libc++ `std::sort` does: LLVM pdqsort, `__algorithm/sort.h` in the macOS SDK. Blender's
/// `IMesh::populate_vert` sorts its `vert_` array with `parallel_sort` (`BLI_sort.hh:11-20`, the TBB
/// build), and on macOS that reached libc++ `std::sort`. The sort is not stable, so equal origins keep
/// the order this routine produces. Blender 5.2.2 matched it on all 106 populate calls of the EXACT
/// corpus (up to 40 vertices) and reproduces every oracle tie edge. The corpus never reaches the heap
/// fallback. That fallback treats the element as non-arithmetic, as Blender's `const Vert *` is.
pub(super) fn libcxx_sort_by<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], less: F) {
    let n = v.len() as isize;
    if n == 0 {
        return;
    }
    let depth = 2 * (n as u64).ilog2() as isize;
    Sorter { v, less }.introsort(0, n, depth, true);
}

/// Blender `parallel_sort` for `IMesh::populate_vert` in a TBB build: `tbb::parallel_sort`
/// (`BLI_sort.hh:11-20`) from oneTBB v2022.3.0, the version Blender 5.2.2 ships (`versions.cmake`).
/// It sorts fewer than 500 elements with `std::sort` (libc++ on macOS, [`libcxx_sort_by`]). For 500
/// or more it runs `parallel_quick_sort`: a pretest, then a recursive split of `quick_sort_range`.
/// The tie order is recorded by `testdata/tbb/record_tbb_sort.cpp` against the v2022.3.0 headers and
/// Blender's `libtbb`. Blender's order depends on the auto partitioner above about 12k elements,
/// which this model does not reproduce.
pub(super) fn parallel_sort_by<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], less: F) {
    if v.len() < TBB_MIN_PARALLEL_SIZE {
        libcxx_sort_by(v, less);
        return;
    }
    let lt = |a: &T, b: &T| less(a, b);
    // `parallel_quick_sort`: a pretest over the first elements, then over the rest. An already
    // sorted input is left unchanged.
    let mut unsorted = (0..TBB_SERIAL_CUTOFF).any(|k| lt(&v[k + 1], &v[k]));
    if !unsorted {
        unsorted = (TBB_SERIAL_CUTOFF + 1..v.len()).any(|k| lt(&v[k], &v[k - 1]));
    }
    if unsorted {
        tbb_quick_sort_range(v, 0, v.len(), &lt);
    }
}

/// oneTBB `parallel_sort` threshold (`min_parallel_size`, `parallel_sort.h`).
const TBB_MIN_PARALLEL_SIZE: usize = 500;
/// oneTBB `parallel_quick_sort` pretest cutoff (`serial_cutoff`, `parallel_sort.h`).
const TBB_SERIAL_CUTOFF: usize = 9;
/// oneTBB `quick_sort_range::grainsize`: ranges below this are sorted by `std::sort` as leaves.
const TBB_GRAINSIZE: usize = 500;

/// oneTBB v2022.3.0 `quick_sort_range` on `v[first..first+size]`, depth-first: `split_range` until a
/// range is below `grainsize`, then libc++ `std::sort` on the leaf. The recursion is complete, so
/// the order does not depend on how many threads ran it.
fn tbb_quick_sort_range<T: Copy>(
    v: &mut [T],
    first: usize,
    size: usize,
    lt: &dyn Fn(&T, &T) -> bool,
) {
    if size < TBB_GRAINSIZE {
        libcxx_sort_by(&mut v[first..first + size], |a, b| lt(a, b));
        return;
    }
    let (left, right_first, right_size) = tbb_split_range(v, first, size, lt);
    tbb_quick_sort_range(v, first, left, lt);
    tbb_quick_sort_range(v, right_first, right_size, lt);
}

/// oneTBB v2022.3.0 `quick_sort_range::split_range`: a two-sided Hoare partition around the
/// pseudo-median of nine. The pivot stays at the range start while the scans run, then moves to its
/// final index `j`. Returns the left size, and the first index and size of the right range. The
/// pivot at `j` belongs to neither range.
fn tbb_split_range<T: Copy>(
    v: &mut [T],
    first: usize,
    size: usize,
    lt: &dyn Fn(&T, &T) -> bool,
) -> (usize, usize, usize) {
    // `pseudo_median_of_nine`
    let offset = size / 8;
    let a = tbb_median_of_three(v, first, 0, offset, offset * 2, lt);
    let b = tbb_median_of_three(v, first, offset * 3, offset * 4, offset * 5, lt);
    let c = tbb_median_of_three(v, first, offset * 6, offset * 7, size - 1, lt);
    let m = tbb_median_of_three(v, first, a, b, c, lt);
    if m != 0 {
        v.swap(first, first + m);
    }
    // `first_element`: index `first` is only written by the final pivot move below.
    let pivot = v[first];
    let mut i = 0usize;
    let mut j = size;
    let pivot_index = 'partition: loop {
        // `do { --j; } while (comp(*first_element, array[j]))`
        loop {
            j -= 1;
            if !lt(&pivot, &v[first + j]) {
                break;
            }
        }
        // `do { if (i == j) goto partition; ++i; } while (comp(array[i], *first_element))`
        loop {
            if i == j {
                break 'partition j;
            }
            i += 1;
            if !lt(&v[first + i], &pivot) {
                break;
            }
        }
        if i == j {
            break 'partition j;
        }
        v.swap(first + i, first + j);
    };
    // `iter_swap(array + j, first_element)`: the pivot takes its final index.
    v.swap(first, first + pivot_index);
    (pivot_index, first + pivot_index + 1, size - pivot_index - 1)
}

/// oneTBB `pseudo_median_of_nine` helper: index (relative to `first`) of the median of three.
fn tbb_median_of_three<T: Copy>(
    v: &[T],
    first: usize,
    l: usize,
    m: usize,
    r: usize,
    lt: &dyn Fn(&T, &T) -> bool,
) -> usize {
    let (vl, vm, vr) = (v[first + l], v[first + m], v[first + r]);
    if lt(&vl, &vm) {
        if lt(&vm, &vr) {
            m
        } else if lt(&vl, &vr) {
            r
        } else {
            l
        }
    } else if lt(&vr, &vm) {
        m
    } else if lt(&vr, &vl) {
        r
    } else {
        l
    }
}


struct Sorter<'a, T, F> {
    v: &'a mut [T],
    less: F,
}

impl<T: Copy, F: Fn(&T, &T) -> bool> Sorter<'_, T, F> {
    fn at(&self, i: isize) -> T {
        self.v[i as usize]
    }

    fn set(&mut self, i: isize, x: T) {
        self.v[i as usize] = x;
    }

    fn swap(&mut self, a: isize, b: isize) {
        self.v.swap(a as usize, b as usize);
    }

    /// `comp(*a, *b)`
    fn lt(&self, a: isize, b: isize) -> bool {
        (self.less)(&self.v[a as usize], &self.v[b as usize])
    }

    /// `comp(*a, p)`
    fn lt_p(&self, a: isize, p: &T) -> bool {
        (self.less)(&self.v[a as usize], p)
    }

    /// `comp(p, *a)`
    fn p_lt(&self, p: &T, a: isize) -> bool {
        (self.less)(p, &self.v[a as usize])
    }

    fn sort3(&mut self, x: isize, y: isize, z: isize) {
        if !self.lt(y, x) {
            if !self.lt(z, y) {
                return;
            }
            self.swap(y, z);
            if self.lt(y, x) {
                self.swap(x, y);
            }
            return;
        }
        if self.lt(z, y) {
            self.swap(x, z);
            return;
        }
        self.swap(x, y);
        if self.lt(z, y) {
            self.swap(y, z);
        }
    }

    fn sort4(&mut self, x1: isize, x2: isize, x3: isize, x4: isize) {
        self.sort3(x1, x2, x3);
        if self.lt(x4, x3) {
            self.swap(x3, x4);
            if self.lt(x3, x2) {
                self.swap(x2, x3);
                if self.lt(x2, x1) {
                    self.swap(x1, x2);
                }
            }
        }
    }

    fn sort5(&mut self, x1: isize, x2: isize, x3: isize, x4: isize, x5: isize) {
        self.sort4(x1, x2, x3, x4);
        if self.lt(x5, x4) {
            self.swap(x4, x5);
            if self.lt(x4, x3) {
                self.swap(x3, x4);
                if self.lt(x3, x2) {
                    self.swap(x2, x3);
                    if self.lt(x2, x1) {
                        self.swap(x1, x2);
                    }
                }
            }
        }
    }

    /// Insertion sort with a guard. Equal to libc++'s unguarded form, because the element before
    /// `first` never compares greater than a value of the range.
    fn insertion_sort(&mut self, first: isize, last: isize) {
        if first == last {
            return;
        }
        let mut i = first + 1;
        while i != last {
            if self.lt(i, i - 1) {
                let t = self.at(i);
                let mut k = i - 1;
                let mut j = i;
                loop {
                    let x = self.at(k);
                    self.set(j, x);
                    j = k;
                    if j == first {
                        break;
                    }
                    k -= 1;
                    if !self.p_lt(&t, k) {
                        break;
                    }
                }
                self.set(j, t);
            }
            i += 1;
        }
    }

    fn insertion_sort_incomplete(&mut self, first: isize, last: isize) -> bool {
        match last - first {
            0 | 1 => true,
            2 => {
                if self.lt(last - 1, first) {
                    self.swap(first, last - 1);
                }
                true
            }
            3 => {
                self.sort3(first, first + 1, last - 1);
                true
            }
            4 => {
                self.sort4(first, first + 1, first + 2, last - 1);
                true
            }
            5 => {
                self.sort5(first, first + 1, first + 2, first + 3, last - 1);
                true
            }
            _ => {
                let limit = 8;
                let mut count = 0;
                let mut j = first + 2;
                self.sort3(first, first + 1, j);
                let mut i = j + 1;
                while i != last {
                    if self.lt(i, j) {
                        let t = self.at(i);
                        let mut k = j;
                        j = i;
                        loop {
                            let x = self.at(k);
                            self.set(j, x);
                            j = k;
                            if j == first {
                                break;
                            }
                            k -= 1;
                            if !self.p_lt(&t, k) {
                                break;
                            }
                        }
                        self.set(j, t);
                        count += 1;
                        if count == limit {
                            i += 1;
                            return i == last;
                        }
                    }
                    j = i;
                    i += 1;
                }
                true
            }
        }
    }

    /// libc++ `__partition_with_equals_on_right`: returns the pivot position and whether the range
    /// was already partitioned.
    fn partition_right(&mut self, mut first: isize, mut last: isize) -> (isize, bool) {
        let begin = first;
        let pivot = self.at(first);
        first += 1;
        while self.lt_p(first, &pivot) {
            first += 1;
        }
        if begin == first - 1 {
            while first < last {
                last -= 1;
                if self.lt_p(last, &pivot) {
                    break;
                }
            }
        } else {
            loop {
                last -= 1;
                if self.lt_p(last, &pivot) {
                    break;
                }
            }
        }
        let already = first >= last;
        while first < last {
            self.swap(first, last);
            loop {
                first += 1;
                if !self.lt_p(first, &pivot) {
                    break;
                }
            }
            loop {
                last -= 1;
                if self.lt_p(last, &pivot) {
                    break;
                }
            }
        }
        let pivot_pos = first - 1;
        if begin != pivot_pos {
            let x = self.at(pivot_pos);
            self.set(begin, x);
        }
        self.set(pivot_pos, pivot);
        (pivot_pos, already)
    }

    /// libc++ `__partition_with_equals_on_left`: returns the first position of the right part.
    fn partition_left(&mut self, mut first: isize, mut last: isize) -> isize {
        let begin = first;
        let pivot = self.at(first);
        if self.p_lt(&pivot, last - 1) {
            loop {
                first += 1;
                if self.p_lt(&pivot, first) {
                    break;
                }
            }
        } else {
            loop {
                first += 1;
                if first >= last || self.p_lt(&pivot, first) {
                    break;
                }
            }
        }
        if first < last {
            loop {
                last -= 1;
                if !self.p_lt(&pivot, last) {
                    break;
                }
            }
        }
        while first < last {
            self.swap(first, last);
            loop {
                first += 1;
                if self.p_lt(&pivot, first) {
                    break;
                }
            }
            loop {
                last -= 1;
                if !self.p_lt(&pivot, last) {
                    break;
                }
            }
        }
        let pivot_pos = first - 1;
        if begin != pivot_pos {
            let x = self.at(pivot_pos);
            self.set(begin, x);
        }
        self.set(pivot_pos, pivot);
        first
    }

    fn introsort(
        &mut self,
        mut first: isize,
        mut last: isize,
        mut depth: isize,
        mut leftmost: bool,
    ) {
        const LIMIT: isize = 24;
        const NINTHER_THRESHOLD: isize = 128;
        loop {
            let len = last - first;
            match len {
                0 | 1 => return,
                2 => {
                    if self.lt(last - 1, first) {
                        self.swap(first, last - 1);
                    }
                    return;
                }
                3 => {
                    self.sort3(first, first + 1, last - 1);
                    return;
                }
                4 => {
                    self.sort4(first, first + 1, first + 2, last - 1);
                    return;
                }
                5 => {
                    self.sort5(first, first + 1, first + 2, first + 3, last - 1);
                    return;
                }
                _ => {}
            }
            if len < LIMIT {
                self.insertion_sort(first, last);
                return;
            }
            if depth == 0 {
                self.partial_sort_all(first, last);
                return;
            }
            depth -= 1;
            let half = len / 2;
            if len > NINTHER_THRESHOLD {
                self.sort3(first, first + half, last - 1);
                self.sort3(first + 1, first + (half - 1), last - 2);
                self.sort3(first + 2, first + (half + 1), last - 3);
                self.sort3(first + (half - 1), first + half, first + (half + 1));
                self.swap(first, first + half);
            } else {
                self.sort3(first + half, first, last - 1);
            }
            if !leftmost && !self.lt(first - 1, first) {
                first = self.partition_left(first, last);
                continue;
            }
            let (i, already) = self.partition_right(first, last);
            if already {
                let fs = self.insertion_sort_incomplete(first, i);
                if self.insertion_sort_incomplete(i + 1, last) {
                    if fs {
                        return;
                    }
                    last = i;
                    continue;
                } else if fs {
                    first = i + 1;
                    continue;
                }
            }
            self.introsort(first, i, depth, leftmost);
            leftmost = false;
            first = i + 1;
        }
    }

    /// libc++ `__partial_sort(first, last, last)`: heap sort of the whole range.
    fn partial_sort_all(&mut self, first: isize, last: isize) {
        let n = last - first;
        if n <= 1 {
            return;
        }
        let mut start = (n - 2) / 2;
        while start >= 0 {
            self.sift_down(first, n, start);
            start -= 1;
        }
        let mut end = last;
        let mut m = n;
        while m > 1 {
            self.pop_heap(first, end, m);
            end -= 1;
            m -= 1;
        }
    }

    fn sift_down(&mut self, first: isize, len: isize, start: isize) {
        let mut child = start;
        if len < 2 || (len - 2) / 2 < child {
            return;
        }
        child = 2 * child + 1;
        if child + 1 < len && self.lt(first + child, first + child + 1) {
            child += 1;
        }
        if self.lt(first + child, first + start) {
            return;
        }
        let mut start = start;
        let top = self.at(first + start);
        loop {
            let x = self.at(first + child);
            self.set(first + start, x);
            start = child;
            if (len - 2) / 2 < child {
                break;
            }
            child = 2 * child + 1;
            if child + 1 < len && self.lt(first + child, first + child + 1) {
                child += 1;
            }
            if self.lt_p(first + child, &top) {
                break;
            }
        }
        self.set(first + start, top);
    }

    /// libc++ `__floyd_sift_down`: returns the hole position.
    fn floyd_sift_down(&mut self, first: isize, len: isize) -> isize {
        let mut hole = first;
        let mut child_i = first;
        let mut child = 0;
        loop {
            child_i += child + 1;
            child = 2 * child + 1;
            if child + 1 < len && self.lt(child_i, child_i + 1) {
                child_i += 1;
                child += 1;
            }
            let x = self.at(child_i);
            self.set(hole, x);
            hole = child_i;
            if child > (len - 2) / 2 {
                return hole;
            }
        }
    }

    fn sift_up(&mut self, first: isize, mut last: isize, mut len: isize) {
        if len > 1 {
            len = (len - 2) / 2;
            let mut ptr = first + len;
            last -= 1;
            if self.lt(ptr, last) {
                let t = self.at(last);
                loop {
                    let x = self.at(ptr);
                    self.set(last, x);
                    last = ptr;
                    if len == 0 {
                        break;
                    }
                    len = (len - 1) / 2;
                    ptr = first + len;
                    if !self.lt_p(ptr, &t) {
                        break;
                    }
                }
                self.set(last, t);
            }
        }
    }

    fn pop_heap(&mut self, first: isize, mut last: isize, len: isize) {
        if len > 1 {
            let top = self.at(first);
            let mut hole = self.floyd_sift_down(first, len);
            last -= 1;
            if hole == last {
                self.set(hole, top);
            } else {
                let x = self.at(last);
                self.set(hole, x);
                hole += 1;
                self.set(last, top);
                self.sift_up(first, hole, hole - first);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Blender's order key for a seen vertex (`plan` in `output.rs`): created vertices (origin -1)
    /// after input vertices, ties on the origin.
    fn key(orig: i32, id: u32) -> (bool, i32, u32) {
        if orig == -1 {
            (true, 0, id)
        } else {
            (false, orig, 0)
        }
    }

    /// Origin pattern of the C++ generator: `((i * 7 + shift) % 9) - 1`, so -1 marks created
    /// vertices and origins repeat.
    fn wavy(n: usize, shift: usize) -> Vec<i32> {
        (0..n).map(|i| ((i * 7 + shift) % 9) as i32 - 1).collect()
    }

    /// Many equal origins: `i % 3`.
    fn triples(n: usize) -> Vec<i32> {
        (0..n).map(|i| (i % 3) as i32).collect()
    }

    /// Permutation of input slots after `libcxx_sort_by`, the call `plan` makes.
    fn sorted(origins: &[i32]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..origins.len()).collect();
        libcxx_sort_by(&mut order, |&a, &b| {
            key(origins[a], a as u32) < key(origins[b], b as u32)
        });
        order
    }

    /// Permutation after the heap fallback alone (`__partial_sort(first, last, last)`).
    fn heap_sorted(origins: &[i32]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..origins.len()).collect();
        let less = |&a: &usize, &b: &usize| key(origins[a], a as u32) < key(origins[b], b as u32);
        let n = order.len() as isize;
        Sorter {
            v: &mut order,
            less,
        }
        .partial_sort_all(0, n);
        order
    }

    /// Expected permutations were recorded from libc++ `std::sort` by a C++ program compiled with
    /// `clang++ -std=c++17 -stdlib=libc++` (Apple clang 21.0.0, macOS SDK libc++), on the same
    /// inputs; the heap cases from libc++ `std::partial_sort(first, last, last)` on the same inputs.
    #[test]
    fn sort_matches_libcxx_std_sort() {
        assert_eq!(sorted(&wavy(2, 0)), [1, 0]);
        assert_eq!(sorted(&wavy(5, 1)), [0, 4, 3, 2, 1]);
        assert_eq!(
            sorted(&wavy(16, 2)),
            [5, 14, 0, 9, 4, 13, 8, 3, 12, 7, 2, 11, 6, 15, 1, 10]
        );
        assert_eq!(
            sorted(&wavy(17, 3)),
            [1, 10, 5, 14, 0, 9, 4, 13, 8, 3, 12, 7, 16, 2, 11, 6, 15]
        );
        assert_eq!(
            sorted(&wavy(31, 4)),
            [
                24, 6, 15, 10, 1, 28, 19, 23, 5, 14, 0, 9, 18, 27, 13, 22, 4, 8, 17, 26, 12, 21, 3,
                30, 16, 7, 25, 2, 11, 20, 29
            ]
        );
        assert_eq!(
            sorted(&wavy(40, 5)),
            [
                2, 38, 29, 11, 20, 6, 33, 24, 15, 19, 1, 37, 10, 28, 5, 32, 14, 23, 0, 18, 27, 9,
                36, 22, 13, 31, 4, 17, 26, 8, 35, 21, 12, 30, 3, 39, 7, 16, 25, 34
            ]
        );
        assert_eq!(
            sorted(&wavy(64, 6)),
            [
                16, 61, 43, 25, 7, 34, 52, 38, 2, 29, 47, 56, 20, 11, 51, 15, 24, 6, 42, 60, 33,
                55, 19, 10, 46, 28, 1, 37, 14, 50, 59, 23, 5, 41, 32, 0, 27, 63, 9, 54, 18, 45, 36,
                22, 49, 40, 13, 31, 58, 4, 26, 44, 17, 53, 8, 62, 35, 3, 12, 21, 30, 39, 48, 57
            ]
        );
        assert_eq!(
            sorted(&triples(40)),
            [
                39, 3, 27, 9, 12, 15, 18, 0, 21, 24, 30, 33, 6, 36, 13, 16, 19, 22, 25, 37, 10, 7,
                34, 4, 31, 1, 28, 8, 26, 29, 23, 17, 32, 14, 11, 35, 5, 2, 38, 20
            ]
        );
    }

    #[test]
    fn heap_fallback_matches_libcxx_partial_sort() {
        assert_eq!(heap_sorted(&wavy(2, 0)), [1, 0]);
        assert_eq!(heap_sorted(&wavy(5, 1)), [0, 4, 3, 2, 1]);
        assert_eq!(
            heap_sorted(&wavy(16, 2)),
            [5, 14, 9, 0, 13, 4, 8, 12, 3, 7, 2, 11, 6, 15, 1, 10]
        );
        assert_eq!(
            heap_sorted(&wavy(17, 3)),
            [10, 1, 5, 14, 9, 0, 13, 4, 8, 12, 3, 16, 7, 2, 11, 6, 15]
        );
        assert_eq!(
            heap_sorted(&wavy(31, 4)),
            [
                6, 24, 15, 1, 28, 19, 10, 14, 23, 5, 0, 9, 18, 27, 13, 22, 4, 26, 8, 17, 30, 12,
                21, 3, 25, 7, 16, 2, 11, 20, 29
            ]
        );
        assert_eq!(
            heap_sorted(&wavy(40, 5)),
            [
                29, 20, 11, 2, 38, 24, 33, 15, 6, 28, 10, 37, 19, 1, 14, 5, 23, 32, 27, 36, 9, 18,
                0, 13, 22, 4, 31, 26, 8, 17, 35, 30, 12, 21, 39, 3, 7, 16, 25, 34
            ]
        );
        assert_eq!(
            heap_sorted(&wavy(64, 6)),
            [
                43, 7, 16, 61, 25, 34, 52, 2, 56, 47, 29, 20, 38, 11, 60, 24, 51, 6, 33, 42, 15, 1,
                46, 10, 55, 19, 37, 28, 14, 59, 50, 23, 41, 5, 32, 27, 54, 0, 45, 9, 18, 36, 63,
                13, 58, 49, 22, 4, 40, 31, 62, 26, 53, 44, 8, 17, 35, 3, 12, 21, 30, 39, 48, 57
            ]
        );
        assert_eq!(
            heap_sorted(&triples(40)),
            [
                18, 36, 33, 24, 0, 21, 9, 39, 15, 27, 6, 30, 12, 3, 13, 28, 25, 22, 10, 4, 19, 37,
                1, 16, 34, 7, 31, 14, 29, 26, 2, 5, 11, 23, 20, 38, 8, 17, 35, 32
            ]
        );
    }

    /// Parses a `keys ...` or `order ...` line of a recording (`testdata/tbb/record_tbb_sort.cpp`).
    fn record_values<T: std::str::FromStr>(line: Option<&str>, tag: &str) -> Vec<T>
    where
        T::Err: std::fmt::Debug,
    {
        let mut fields = line.expect("recording line").split_whitespace();
        assert_eq!(fields.next(), Some(tag));
        fields
            .map(|field| field.parse().expect("recorded integer"))
            .collect()
    }

    /// Sorts each recorded case with `parallel_sort_by` and compares the element order with the
    /// `tbb::parallel_sort` run of oneTBB v2022.3.0 against Blender's `libtbb`. Five cases give a
    /// different order under oneTBB master's `parallel_sort`, so the test rejects that version.
    #[test]
    fn parallel_sort_matches_recorded_tbb_v2022_3_0_order() {
        let text = include_str!("testdata/tbb/tbb_sort_v2022_3_0.txt");
        let mut lines = text.lines().filter(|line| !line.starts_with('#'));
        let mut cases = 0;
        while let Some(header) = lines.next() {
            let mut fields = header.split_whitespace();
            assert_eq!(fields.next(), Some("case"));
            let name = fields.next().expect("case name").to_owned();
            let n: usize = fields.next().expect("case size").parse().expect("case size");
            let keys: Vec<i64> = record_values(lines.next(), "keys");
            let order: Vec<usize> = record_values(lines.next(), "order");
            assert_eq!(keys.len(), n, "{name}: key count");
            assert_eq!(order.len(), n, "{name}: order count");
            let mut sorted: Vec<usize> = (0..n).collect();
            parallel_sort_by(&mut sorted, |&a, &b| keys[a] < keys[b]);
            assert_eq!(sorted, order, "{name}: order differs from tbb::parallel_sort v2022.3.0");
            cases += 1;
        }
        assert_eq!(cases, 9, "the recording holds nine cases");
    }
}
