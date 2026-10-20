//! Slot order of Blender's `blender::Map` (`blenlib/BLI_map.hh`), for the maps whose iteration
//! order decides output order.
//!
//! Blender iterates a `Map` over its slot array, so the order of `items()` is a function of the
//! hash values, the reserve calls, and the insertion sequence. Mesh Arrangements uses such maps
//! in `find_clusters` (`mesh_intersect.cc`): `itt_map` (`reserve(overlap size)`, then `add_new` in
//! overlap order) and `plane_cls` (`reserve(maybe_coplanar_tris.size())`). Its hashes are value
//! hashes, not addresses, so the order is reproducible.
//!
//! Reproduced exactly:
//! - Load factor `LOAD_FACTOR 1, 2` (`BLI_map.hh`): usable slots are half the total.
//! - Total slots: `power_of_2_max(ceil(min_usable * 2))` (`BLI_hash_tables.hh`
//!   `total_slot_amount_for_usable_slots`), where `min_usable = max(n, 2 * size)` (`realloc_and_reinsert`).
//! - Probing: `DefaultProbingStrategy = PythonProbingStrategy<>` (`BLI_probing_strategies.hh`): one
//!   slot per step, `perturb >>= 5`, `hash = 5 * hash + 1 + perturb`.
//! - Growth reinserts the occupied slots in old index order (`add_after_grow`).
//! - Hashes: `get_default_hash(int, int)` for pairs (`BLI_hash.hh`), and `Plane::hash()` for
//!   planes (`mesh_intersect.cc`, in `intersect/cluster.rs`).
//!
//! Not modelled: removal. None of the maps emulated here are ever shrunk, so no slot is ever
//! marked removed.

/// Blender `blenlib_detail::default_hash_factors[0]`, the multiplier for the second element of a
/// pair (`BLI_hash.hh`).
const PAIR_SECOND_FACTOR: u64 = 19_349_669;

/// Blender `get_default_hash(int a, int b)` for `std::pair<int, int>`: `DefaultHash<int>(a) ^
/// (factor * DefaultHash<int>(b))`, with `DefaultHash<int>(v) = uint64(v)` (`BLI_hash.hh`).
pub(super) fn pair_hash(a: usize, b: usize) -> u64 {
    (a as u64) ^ PAIR_SECOND_FACTOR.wrapping_mul(b as u64)
}

struct Entry<K, V> {
    key: K,
    hash: u64,
    value: V,
}

/// Slot array with Blender's slot order. Slots keep insertion-probed positions, so `items` is
/// Blender's `Map::items()` order.
pub(super) struct BlenderMap<K, V> {
    slots: Vec<Option<Entry<K, V>>>,
    /// Occupied slots (`Map::size`). No removals, so this is also `occupied_and_removed`.
    size: usize,
    /// Slots usable before a grow (`usable_slots_`).
    usable: usize,
}

impl<K: PartialEq, V> BlenderMap<K, V> {
    pub(super) fn new() -> Self {
        Self {
            slots: Vec::new(),
            size: 0,
            usable: 0,
        }
    }

    /// Blender `Map::reserve`: grows so that `n` entries fit without another grow.
    pub(super) fn reserve(&mut self, n: usize) {
        if self.usable < n {
            self.realloc(n);
        }
    }

    /// Blender `Map::add_new`: the key must be absent.
    pub(super) fn add_new(&mut self, key: K, hash: u64, value: V) {
        if self.size >= self.usable {
            self.realloc(self.size + 1);
        }
        let mask = self.slots.len() - 1;
        let slot = Self::probe_empty(&self.slots, hash, mask);
        self.slots[slot] = Some(Entry { key, hash, value });
        self.size += 1;
    }

    /// Blender `Map::lookup_ptr`: the value stored under `key`, if any.
    pub(super) fn get_mut(&mut self, key: &K, hash: u64) -> Option<&mut V> {
        let mask = self.slots.len().checked_sub(1)?;
        let mut probe = Probe::new(hash);
        loop {
            let index = (probe.get() as usize) & mask;
            match &self.slots[index] {
                None => return None,
                Some(e) if e.hash == hash && e.key == *key => break,
                Some(_) => probe.next(),
            }
        }
        let index = (probe.get() as usize) & mask;
        self.slots[index].as_mut().map(|e| &mut e.value)
    }

    /// Blender `Map::items()`: entries in slot-index order.
    pub(super) fn into_items(self) -> impl Iterator<Item = (K, V)> {
        self.slots.into_iter().flatten().map(|e| (e.key, e.value))
    }

    /// Blender `realloc_and_reinsert(n)`.
    fn realloc(&mut self, n: usize) {
        let min_usable = n.max(self.size * 2);
        let total = (min_usable * 2).next_power_of_two();
        let usable = total / 2;
        let mut new_slots: Vec<Option<Entry<K, V>>> = Vec::with_capacity(total);
        new_slots.resize_with(total, || None);
        let mask = total - 1;
        for entry in std::mem::take(&mut self.slots).into_iter().flatten() {
            let slot = Self::probe_empty(&new_slots, entry.hash, mask);
            new_slots[slot] = Some(entry);
        }
        self.slots = new_slots;
        self.usable = usable;
    }

    /// First empty slot along the probe sequence of `hash` (`add_after_grow`, `add_new__impl`).
    fn probe_empty(slots: &[Option<Entry<K, V>>], hash: u64, mask: usize) -> usize {
        let mut probe = Probe::new(hash);
        loop {
            let index = (probe.get() as usize) & mask;
            if slots[index].is_none() {
                return index;
            }
            probe.next();
        }
    }
}

/// Blender `PythonProbingStrategy<1, false>` (`BLI_probing_strategies.hh`), the default strategy.
struct Probe {
    hash: u64,
    perturb: u64,
}

impl Probe {
    fn new(hash: u64) -> Self {
        Self {
            hash,
            perturb: hash,
        }
    }

    fn get(&self) -> u64 {
        self.hash
    }

    fn next(&mut self) {
        self.perturb >>= 5;
        self.hash = self
            .hash
            .wrapping_mul(5)
            .wrapping_add(1)
            .wrapping_add(self.perturb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plane-map inserts recorded from a Blender 5.2.2 debug build (`SimpleMapSlot<Plane>::
    /// occupy_no_value`, hash argument), in insertion order. The keys are the quads
    /// `[9,6,3,0,10,7,4,1,11,8,5,2]`.
    const RECORDED_PLANE_HASHES: [u64; 12] = [
        18_444_492_273_897_568_511,
        9_223_372_036_856_087_552,
        18_442_240_474_083_883_519,
        4_607_182_418_800_279_551,
        9_223_372_036_856_284_160,
        18_444_492_273_897_699_583,
        4_607_182_418_800_082_943,
        18_442_240_474_084_014_591,
        18_444_492_273_897_502_975,
        9_223_372_036_856_218_624,
        18_442_240_474_083_817_983,
        4_607_182_418_800_148_479,
    ];
    const RECORDED_INSERT_KEYS: [usize; 12] = [9, 6, 3, 0, 10, 7, 4, 1, 11, 8, 5, 2];
    /// Recorded cluster order: `find_clusters` numbers clusters by `plane_cls.items()`, and this
    /// is the plane order of the observed `calc_cluster_tris` blocks (plane per block, decoded
    /// from the CDT output coordinates).
    const RECORDED_CLUSTER_ORDER: [usize; 12] = [6, 5, 7, 2, 3, 11, 10, 4, 8, 1, 0, 9];

    #[test]
    fn reserved_map_reproduces_recorded_blender_slot_order() {
        // `plane_cls.reserve(24)` (24 maybe-coplanar triangles), then `add_new` per plane.
        let mut map: BlenderMap<usize, ()> = BlenderMap::new();
        map.reserve(24);
        for (&key, &hash) in RECORDED_INSERT_KEYS
            .iter()
            .zip(RECORDED_PLANE_HASHES.iter())
        {
            map.add_new(key, hash, ());
        }
        let order: Vec<usize> = map.into_items().map(|(k, ())| k).collect();
        // The recorded Blender slot order of the planes, which is the cluster numbering.
        assert_eq!(
            order.len(),
            12,
            "every recorded plane is present exactly once"
        );
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..12).collect::<Vec<_>>());
        assert_eq!(order, RECORDED_CLUSTER_ORDER.to_vec());
    }

    #[test]
    fn lookup_finds_inserted_keys_through_probe_chains() {
        let mut map: BlenderMap<usize, u32> = BlenderMap::new();
        map.reserve(24);
        for (&key, &hash) in RECORDED_INSERT_KEYS
            .iter()
            .zip(RECORDED_PLANE_HASHES.iter())
        {
            map.add_new(key, hash, key as u32 * 10);
        }
        for (&key, &hash) in RECORDED_INSERT_KEYS
            .iter()
            .zip(RECORDED_PLANE_HASHES.iter())
        {
            assert_eq!(map.get_mut(&key, hash).copied(), Some(key as u32 * 10));
        }
        assert_eq!(map.get_mut(&99, RECORDED_PLANE_HASHES[0]), None);
    }

    #[test]
    fn pair_hash_matches_blender_default_hash() {
        // get_default_hash(a, b) = a ^ (19349669 * b)
        assert_eq!(pair_hash(0, 1), 19_349_669);
        assert_eq!(pair_hash(2, 16), 2 ^ (19_349_669 * 16));
    }
}
