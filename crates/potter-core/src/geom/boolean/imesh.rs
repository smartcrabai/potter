//! Shared data model of the Mesh Arrangements port (Blender `meshintersect::Vert`, `Face`,
//! `IMeshArena`). Vertices are owned by an [`Arena`] that de-duplicates them by exact position.
//! Triangles refer to vertices by arena index and carry their original indices.

use std::collections::HashMap;

use glam::DVec3;

#[cfg(test)]
use super::exact::rat3_from_dvec3;
use super::exact::{Rat3, rat_to_f64};

/// Blender `NO_INDEX`: "no original element" (created by the algorithm).
pub const NO_INDEX: i32 = -1;

/// A vertex with exact coordinates and a double approximation for float filters.
#[derive(Clone, Debug)]
pub struct Vert {
    /// Exact position.
    pub co: Rat3,
    /// Nearest-double approximation of `co`, used only by float filters.
    pub co_f: DVec3,
    /// Input vertex index this vertex came from, or [`NO_INDEX`] for created vertices.
    pub orig: i32,
}

/// Owns every vertex of one boolean run and merges vertices with equal exact positions.
#[derive(Debug, Default)]
pub struct Arena {
    verts: Vec<Vert>,
    by_co: HashMap<Rat3, u32>,
}

impl Arena {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn vert(&self, id: u32) -> &Vert {
        &self.verts[id as usize]
    }

    #[cfg(test)]
    #[must_use]
    pub fn num_verts(&self) -> usize {
        self.verts.len()
    }

    /// Blender `IMeshArena::add_or_find_vert`: returns the existing vertex with the same exact
    /// position (ignoring `orig`), or allocates one with index equal to its creation order.
    pub fn add_or_find_vert(&mut self, co: Rat3, orig: i32) -> u32 {
        if let Some(&id) = self.by_co.get(&co) {
            return id;
        }
        let id = u32::try_from(self.verts.len()).unwrap_or(u32::MAX);
        let co_f = DVec3::new(rat_to_f64(&co[0]), rat_to_f64(&co[1]), rat_to_f64(&co[2]));
        self.by_co.insert(co.clone(), id);
        self.verts.push(Vert { co, co_f, orig });
        id
    }

    /// Adds a vertex from its double position (exact value of the double).
    #[cfg(test)]
    pub fn add_or_find_dvec(&mut self, co: DVec3, orig: i32) -> u32 {
        self.add_or_find_vert(rat3_from_dvec3(co), orig)
    }
}

/// A triangle of the intersection mesh. Blender `Face` restricted to three vertices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tri {
    /// Arena indices, counter-clockwise around the normal.
    pub v: [u32; 3],
    /// Input face index this triangle derives from (global over joined operands).
    pub orig: i32,
    /// Per edge `k` (from `v[k]` to `v[k+1]`): original input edge index or [`NO_INDEX`].
    pub edge_orig: [i32; 3],
    /// Per edge: whether it was created by intersecting with another face.
    pub is_intersect: [bool; 3],
}

impl Tri {
    #[must_use]
    pub fn new(v: [u32; 3], orig: i32, edge_orig: [i32; 3]) -> Self {
        Self {
            v,
            orig,
            edge_orig,
            is_intersect: [false; 3],
        }
    }
}
