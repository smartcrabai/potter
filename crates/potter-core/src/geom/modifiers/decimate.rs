// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/bmesh/tools/bmesh_decimate_collapse.cc, source/blender/bmesh/tools/bmesh_decimate_dissolve.cc, source/blender/bmesh/tools/bmesh_decimate_unsubdivide.cc, source/blender/bmesh/intern/bmesh_polygon.cc, source/blender/bmesh/operators/bmo_dissolve.cc, SPDX-FileCopyrightText: 2023 Blender Authors, GPL-2.0-or-later.
//   source/blender/modifiers/intern/MOD_decimate.cc, SPDX-FileCopyrightText: 2005 Blender Authors, GPL-2.0-or-later.

use std::collections::{HashMap, HashSet};

use glam::DVec3;
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{provenance::Origin, uv_layers},
    model::Modifier,
};

use super::subdivision_uv::carried_pins;
use super::{
    Mesh, Polygon, bool_param, cyclic_pairs, decimate_compact, edge_key, id_indices,
    invalid_parameter, number_param, positions, schema_value,
};
mod collapse;
mod heap;
mod interp;
mod kdtree;
mod normals;
mod polyfill;
mod quad;

pub(super) use collapse::decimate_collapse;

pub(super) fn decimate_face_normal(points: &[DVec3], face: &[usize]) -> DVec3 {
    let mut normal = DVec3::ZERO;
    for (first, second) in cyclic_pairs(face) {
        normal += points[*first].cross(points[*second]);
    }
    normal.normalize_or_zero()
}

/// One `[u, v]` per UV layer, in the collapse's layer order.
type CornerUv = Vec<[f64; 2]>;

/// UV layers of `mesh` as `(names, values)`: `values[l][c]` is the input `[u, v]` of corner `c` (face
/// order, then corner order) in layer `names[l]`, from [`uv_layers::read_layers`]. A corner without a
/// value takes `[0, 0]`.
fn collapse_uv_layers(mesh: &Mesh) -> (Vec<String>, Vec<Vec<[f64; 2]>>) {
    let mut names = Vec::new();
    let mut values = Vec::new();
    for layer in uv_layers::read_layers(mesh) {
        names.push(layer.name);
        values.push(
            layer
                .faces
                .iter()
                .zip(&mesh.faces)
                .flat_map(|(uv, face)| {
                    (0..face.vertices.len())
                        .map(move |corner| uv.as_ref().map_or([0.0, 0.0], |uv| uv[corner]))
                })
                .collect(),
        );
    }
    (names, values)
}

/// Rewrites the UV layers of `mesh`, rebuilt by a collapse, from the per-corner values of `corner_uv`
/// (output face order, then corner order). `names[l]` is the layer of `corner_uv[c][l]`.
fn collapse_write_uv(mesh: &mut Mesh, names: &[String], corner_uv: &[CornerUv]) -> Result<()> {
    let corner_total: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    if corner_total != corner_uv.len() {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "decimate UV values do not match the rebuilt mesh",
        ));
    }
    // The rebuild through provenance carried each corner's pin; the new values keep those pins.
    let carried = uv_layers::read_layers(mesh);
    let layers: Vec<uv_layers::UvLayer> = names
        .iter()
        .enumerate()
        .map(|(l, name)| {
            let mut corner = 0;
            let faces: Vec<Option<Vec<[f64; 2]>>> = mesh
                .faces
                .iter()
                .map(|face| {
                    let row = (corner..corner + face.vertices.len())
                        .map(|c| corner_uv[c][l])
                        .collect();
                    corner += face.vertices.len();
                    Some(row)
                })
                .collect();
            let pinned = carried_pins(&carried, name, &faces);
            let mut layer = uv_layers::UvLayer::new(name.clone(), faces);
            layer.pinned = pinned;
            layer
        })
        .collect();
    uv_layers::write_layers(mesh, &layers);
    Ok(())
}

fn decimate_symmetry_axis(modifier: &Modifier) -> Result<usize> {
    match schema_value(modifier, "symmetry_axis")? {
        Value::String(axis) => match axis.to_ascii_uppercase().as_str() {
            "X" => Ok(0),
            "Y" => Ok(1),
            "Z" => Ok(2),
            _ => Err(invalid_parameter(modifier, "symmetry_axis", "X, Y, or Z")),
        },
        Value::Number(axis) => axis
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value < 3)
            .ok_or_else(|| invalid_parameter(modifier, "symmetry_axis", "X, Y, or Z")),
        _ => Err(invalid_parameter(modifier, "symmetry_axis", "X, Y, or Z")),
    }
}

fn decimate_group_weights(mesh: &Mesh, modifier: &Modifier) -> Result<Option<Vec<f64>>> {
    let group_name = match schema_value(modifier, "vertex_group")? {
        Value::String(name) => name.as_str(),
        _ => {
            return Err(invalid_parameter(
                modifier,
                "vertex_group",
                "a vertex-group name",
            ));
        }
    };
    if group_name.is_empty() {
        return Ok(None);
    }
    let weights = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .and_then(|groups| groups.get(group_name))
        .and_then(Value::as_object)
        .ok_or_else(|| {
            invalid_parameter(modifier, "vertex_group", "a group present on the mesh")
        })?;
    let invert = bool_param(modifier, "invert_vertex_group")?;
    mesh.vertices
        .iter()
        .map(|vertex| {
            let weight = weights
                .get(&format!("v{}", vertex.id))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(invalid_parameter(
                    modifier,
                    "vertex_group",
                    "finite weights from 0 through 1",
                ));
            }
            Ok(if invert { 1.0 - weight } else { weight })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

// Blender's `BM_mesh_decimate_dissolve` and `BM_mesh_decimate_unsubdivide` run on a BMesh. `BMesh`
// below keeps the disk and radial cycles of `collapse.rs`, with arena indices in creation order
// (Blender's element list order), and the operations these two decimators use: `BM_faces_join`,
// `BM_face_split`, `BM_vert_collapse_edge`, `BM_vert_dissolve` and their kernels.

/// Blender's `NONE` link: an absent element index.
const NONE: usize = usize::MAX;
/// Blender's `COST_INVALID`.
const COST_INVALID: f32 = f32::MAX;
/// Blender's `VERT_DISSOLVE_MAX`: the most edges or faces around a dissolvable vertex.
const VERT_DISSOLVE_MAX: usize = 4;
/// Blender's `VERT_INDEX_DO_COLLAPSE`, `VERT_INDEX_INIT` and `VERT_INDEX_IGNORE`.
const TAG_DO_COLLAPSE: i8 = -1;
const TAG_INIT: i8 = 0;
const TAG_IGNORE: i8 = 1;
/// `DEG2RADF(90.0f)`: the unit of the vertex and edge angle products, and their fallback angle.
const UNIT_TO_ANGLE: f32 = 90.0 * (std::f64::consts::PI / 180.0) as f32;
const ANGLE_TO_UNIT: f32 = 1.0 / UNIT_TO_ANGLE;

#[derive(Clone, Copy)]
struct DiskLink {
    next: usize,
    prev: usize,
}

const NO_DISK: DiskLink = DiskLink {
    next: NONE,
    prev: NONE,
};

/// Blender's `BLI_mempool` slot order for one element type. `BLI_mempool_alloc` takes the head of the
/// free list, the most recently freed slot, and then never-used slots in order. Blender emits elements
/// in pool order (`BM_mesh_bm_to_me`), so an element's slot is its output position.
#[derive(Default)]
struct PoolSlots {
    fresh: usize,
    freed: Vec<usize>,
}

impl PoolSlots {
    fn alloc(&mut self) -> usize {
        if let Some(slot) = self.freed.pop() {
            return slot;
        }
        self.fresh += 1;
        self.fresh - 1
    }

    fn release(&mut self, slot: usize) {
        self.freed.push(slot);
    }
}

struct BVert {
    co: [f32; 3],
    /// First edge of the disk cycle, or `NONE`.
    edge: usize,
    alive: bool,
    /// `_FLAG_JF` of `BM_faces_join`.
    jf: bool,
}

#[derive(Clone, Copy)]
struct BEdge {
    v: [usize; 2],
    disk: [DiskLink; 2],
    /// First loop of the radial cycle, or `NONE`.
    l: usize,
    alive: bool,
    /// `BM_ELEM_SMOOTH` is set when the edge is not sharp.
    smooth: bool,
    seam: bool,
    /// `BM_ELEM_TAG`: set on the edges that were already wire before dissolving.
    tag: bool,
    jf: bool,
    /// `BLI_mempool` slot of this edge (see [`PoolSlots`]).
    slot: usize,
}

#[derive(Clone)]
struct BLoop {
    v: usize,
    e: usize,
    f: usize,
    next: usize,
    prev: usize,
    radial_next: usize,
    radial_prev: usize,
    /// One `[u, v]` per UV layer.
    uv: Vec<[f32; 2]>,
    /// Input corner index (flattened over the input faces) the loop's data came from, or `NONE`.
    src: usize,
}

#[derive(Clone)]
struct BFace {
    /// First loop of the face ring, `BM_FACE_FIRST_LOOP`.
    first: usize,
    len: usize,
    no: [f32; 3],
    mat: u32,
    /// Input polygon index, or `None` for a face made by a join or split.
    origin: Option<usize>,
    alive: bool,
    jf: bool,
    /// `BLI_mempool` slot of this face (see [`PoolSlots`]).
    slot: usize,
}

/// The `delimit` modes of the DISSOLVE decimator.
#[derive(Clone, Copy, Default)]
struct Delimit {
    seam: bool,
    sharp: bool,
    material: bool,
    normal: bool,
    uv: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DissolveMethod {
    InteriorFan,
    InteriorChain,
    BoundaryFan,
    WireChain,
}

/// Blender's `BMesh` as the decimators use it. Elements live in arenas indexed by creation order; a
/// killed element is marked dead and never reused. Killed loops are the exception: their slots go
/// on `free_loops` and are reused. No output depends on loop indices, since faces are walked from
/// their `first` loop; without reuse a join's new face would keep every ring it ever replaced.
struct BMesh {
    verts: Vec<BVert>,
    edges: Vec<BEdge>,
    loops: Vec<BLoop>,
    faces: Vec<BFace>,
    /// Loop slots of killed faces, available to `loop_alloc`.
    free_loops: Vec<usize>,
    /// `BLI_mempool` slots of the edges and faces: the output order of the live elements.
    edge_slots: PoolSlots,
    face_slots: PoolSlots,
    /// Scratch membership of `sort_winding`: an edge (vertex) is in the set when its mark equals
    /// `mark_stamp`, so starting a new set costs nothing.
    edge_mark: Vec<usize>,
    vert_mark: Vec<usize>,
    mark_stamp: usize,
    /// Scratch rings of `face_create_ngon`, kept for their capacity.
    ring_verts: Vec<usize>,
    ring_edges: Vec<usize>,
    uv_layers: usize,
}

impl BMesh {
    fn disk_index(&self, e: usize, v: usize) -> usize {
        usize::from(self.edges[e].v[1] == v)
    }

    fn disk_next(&self, e: usize, v: usize) -> usize {
        self.edges[e].disk[self.disk_index(e, v)].next
    }

    /// `BM_EDGES_OF_VERT`: the disk cycle of `v`, starting at `v->e`.
    fn disk_edges(&self, v: usize) -> Vec<usize> {
        let first = self.verts[v].edge;
        let mut out = Vec::new();
        if first == NONE {
            return out;
        }
        let mut e = first;
        loop {
            out.push(e);
            e = self.disk_next(e, v);
            if e == first {
                return out;
            }
        }
    }

    /// `BM_FACES_OF_EDGE`'s loops: the radial cycle of `e`, starting at `e->l`.
    fn radial_loops(&self, e: usize) -> Vec<usize> {
        let first = self.edges[e].l;
        let mut out = Vec::new();
        if first == NONE {
            return out;
        }
        let mut l = first;
        loop {
            out.push(l);
            l = self.loops[l].radial_next;
            if l == first {
                return out;
            }
        }
    }

    /// The loops of face `f` from its first loop, without collecting them.
    fn face_loop_iter(&self, f: usize) -> impl Iterator<Item = usize> + '_ {
        let first = self.faces[f].first;
        let mut next = Some(first);
        std::iter::from_fn(move || {
            let l = next?;
            let after = self.loops[l].next;
            next = (after != first).then_some(after);
            Some(l)
        })
    }

    fn face_loops(&self, f: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.faces[f].len);
        out.extend(self.face_loop_iter(f));
        out
    }

    fn other_vert(&self, e: usize, v: usize) -> usize {
        let [a, b] = self.edges[e].v;
        if a == v { b } else { a }
    }

    fn vert_in_edge(&self, e: usize, v: usize) -> bool {
        self.edges[e].v.contains(&v)
    }

    fn is_wire(&self, e: usize) -> bool {
        self.edges[e].l == NONE
    }

    fn is_boundary(&self, e: usize) -> bool {
        let l = self.edges[e].l;
        l != NONE && self.loops[l].radial_next == l
    }

    fn is_manifold(&self, e: usize) -> bool {
        let l = self.edges[e].l;
        l != NONE
            && self.loops[l].radial_next != l
            && self.loops[self.loops[l].radial_next].radial_next == l
    }

    /// `BM_edge_is_contiguous`.
    fn is_contiguous(&self, e: usize) -> bool {
        let l = self.edges[e].l;
        if l == NONE {
            return false;
        }
        let other = self.loops[l].radial_next;
        other != l && self.loops[other].radial_next == l && self.loops[other].v != self.loops[l].v
    }

    /// `BM_edge_is_contiguous_loop_cd` for one UV layer.
    #[expect(clippy::float_cmp, reason = "Blender compares UV values exactly")]
    fn is_contiguous_uv(&self, e: usize, layer: usize) -> bool {
        let l = self.edges[e].l;
        if l == NONE || self.loops[l].radial_next == l {
            return true;
        }
        let base_next = self.loops[l].next;
        let mut it = self.loops[l].radial_next;
        loop {
            let (a, b) = if self.loops[it].v == self.loops[l].v {
                (it, self.loops[it].next)
            } else {
                (self.loops[it].next, it)
            };
            if self.loops[l].uv[layer] != self.loops[a].uv[layer]
                || self.loops[base_next].uv[layer] != self.loops[b].uv[layer]
            {
                return false;
            }
            it = self.loops[it].radial_next;
            if it == l {
                return true;
            }
        }
    }

    /// `BM_edge_face_count_is_over`.
    fn edge_face_count_over(&self, e: usize, max: usize) -> bool {
        self.radial_loops(e).len() > max
    }

    /// `bmesh_disk_count_at_most`.
    fn disk_count_at_most(&self, v: usize, max: usize) -> usize {
        self.disk_edges(v).len().min(max)
    }

    /// `bmesh_disk_edge_append`: appends `e` at the tail of `v`'s disk cycle.
    fn disk_edge_append(&mut self, e: usize, v: usize) {
        let idx = self.disk_index(e, v);
        let head = self.verts[v].edge;
        if head == NONE {
            self.verts[v].edge = e;
            self.edges[e].disk[idx] = DiskLink { next: e, prev: e };
            return;
        }
        let head_idx = self.disk_index(head, v);
        let tail = self.edges[head].disk[head_idx].prev;
        self.edges[e].disk[idx] = DiskLink {
            next: head,
            prev: tail,
        };
        self.edges[head].disk[head_idx].prev = e;
        let tail_idx = self.disk_index(tail, v);
        self.edges[tail].disk[tail_idx].next = e;
    }

    /// `bmesh_disk_edge_remove`.
    fn disk_edge_remove(&mut self, e: usize, v: usize) {
        let idx = self.disk_index(e, v);
        let DiskLink { next, prev } = self.edges[e].disk[idx];
        if prev != NONE {
            let i = self.disk_index(prev, v);
            self.edges[prev].disk[i].next = next;
        }
        if next != NONE {
            let i = self.disk_index(next, v);
            self.edges[next].disk[i].prev = prev;
        }
        if self.verts[v].edge == e {
            self.verts[v].edge = if e == next { NONE } else { next };
        }
        self.edges[e].disk[idx] = NO_DISK;
    }

    /// `bmesh_radial_loop_append`: the appended loop becomes the head of the radial cycle.
    fn radial_append(&mut self, e: usize, l: usize) {
        let head = self.edges[e].l;
        if head == NONE {
            self.edges[e].l = l;
            self.loops[l].radial_next = l;
            self.loops[l].radial_prev = l;
        } else {
            let after = self.loops[head].radial_next;
            self.loops[l].radial_prev = head;
            self.loops[l].radial_next = after;
            self.loops[after].radial_prev = l;
            self.loops[head].radial_next = l;
            self.edges[e].l = l;
        }
        self.loops[l].e = e;
    }

    /// `bmesh_radial_loop_remove`.
    fn radial_remove(&mut self, e: usize, l: usize) {
        if self.loops[l].radial_next != l {
            if l == self.edges[e].l {
                self.edges[e].l = self.loops[l].radial_next;
            }
            let (next, prev) = (self.loops[l].radial_next, self.loops[l].radial_prev);
            self.loops[next].radial_prev = prev;
            self.loops[prev].radial_next = next;
        } else if l == self.edges[e].l {
            self.edges[e].l = NONE;
        }
        self.loops[l].radial_next = NONE;
        self.loops[l].radial_prev = NONE;
        self.loops[l].e = NONE;
    }

    /// `bmesh_disk_vert_replace`: moves edge `e` from `src` to `dst` in the disk cycles.
    fn disk_vert_replace(&mut self, e: usize, dst: usize, src: usize) {
        self.disk_edge_remove(e, src);
        let idx = self.disk_index(e, src);
        self.edges[e].v[idx] = dst;
        self.edges[e].disk[idx] = NO_DISK;
        self.disk_edge_append(e, dst);
    }

    /// `BM_edge_create`; with `no_double` an existing edge between the vertices is returned.
    fn edge_create(&mut self, v1: usize, v2: usize, no_double: bool) -> usize {
        if no_double && let Some(e) = self.edge_exists(v1, v2) {
            return e;
        }
        let e = self.edges.len();
        let slot = self.edge_slots.alloc();
        self.edges.push(BEdge {
            v: [v1, v2],
            disk: [NO_DISK; 2],
            l: NONE,
            alive: true,
            smooth: false,
            seam: false,
            tag: false,
            jf: false,
            slot,
        });
        self.disk_edge_append(e, v1);
        self.disk_edge_append(e, v2);
        e
    }

    /// `BM_edge_exists`.
    fn edge_exists(&self, v1: usize, v2: usize) -> Option<usize> {
        let first = self.verts[v1].edge;
        if first == NONE {
            return None;
        }
        let mut e = first;
        loop {
            if self.vert_in_edge(e, v2) {
                return Some(e);
            }
            e = self.disk_next(e, v1);
            if e == first {
                return None;
            }
        }
    }

    /// Takes a slot for a new loop of vertex `v`, edge `e` and face `f`: a killed loop's slot when
    /// one is free (keeping its UV storage), else a new arena slot. Only `v`, `e` and `f` are set;
    /// the caller sets the UV and provenance.
    fn loop_alloc(&mut self, v: usize, e: usize, f: usize) -> usize {
        let Some(l) = self.free_loops.pop() else {
            self.loops.push(BLoop {
                v,
                e,
                f,
                next: NONE,
                prev: NONE,
                radial_next: NONE,
                radial_prev: NONE,
                uv: Vec::new(),
                src: NONE,
            });
            return self.loops.len() - 1;
        };
        let lp = &mut self.loops[l];
        lp.v = v;
        lp.e = e;
        lp.f = f;
        lp.next = NONE;
        lp.prev = NONE;
        lp.radial_next = NONE;
        lp.radial_prev = NONE;
        lp.uv.clear();
        lp.src = NONE;
        l
    }

    /// `bm_loop_create` with `example` as the loop whose data is copied.
    fn loop_create(&mut self, v: usize, e: usize, f: usize, example: usize) -> usize {
        let l = self.loop_alloc(v, e, f);
        self.loops[l].src = self.loops[example].src;
        self.copy_loop_uv(l, example);
        l
    }

    /// Copies the UV data of loop `from` into loop `to`, reusing the storage of `to`.
    fn copy_loop_uv(&mut self, to: usize, from: usize) {
        let mut uv = std::mem::take(&mut self.loops[to].uv);
        uv.clear();
        uv.extend_from_slice(&self.loops[from].uv);
        self.loops[to].uv = uv;
    }

    /// `BM_face_create`: one loop per corner, each radially appended to its edge. The face takes the
    /// material and provenance of `example`; its normal starts zeroed, as Blender's new faces do.
    fn face_create(&mut self, verts: &[usize], edges: &[usize], example: usize) -> usize {
        let f = self.faces.len();
        let n = verts.len();
        let mut first = NONE;
        let mut prev = NONE;
        for k in 0..n {
            let l = self.loop_alloc(verts[k], edges[k], f);
            self.loops[l].uv.resize(self.uv_layers, [0.0; 2]);
            self.radial_append(edges[k], l);
            if prev == NONE {
                first = l;
            } else {
                self.loops[prev].next = l;
                self.loops[l].prev = prev;
            }
            prev = l;
        }
        self.loops[prev].next = first;
        self.loops[first].prev = prev;
        let slot = self.face_slots.alloc();
        let source = &self.faces[example];
        let face = BFace {
            first,
            len: n,
            no: [0.0; 3],
            mat: source.mat,
            origin: source.origin,
            alive: true,
            jf: false,
            slot,
        };
        self.faces.push(face);
        f
    }

    /// `BM_face_kill`: removes the face's loops from their radial cycles and frees their slots.
    fn face_kill(&mut self, f: usize) {
        assert!(self.faces[f].alive, "BMesh face killed twice");
        let first = self.faces[f].first;
        let mut l = first;
        loop {
            let next = self.loops[l].next;
            let e = self.loops[l].e;
            self.radial_remove(e, l);
            self.free_loops.push(l);
            l = next;
            if l == first {
                break;
            }
        }
        self.faces[f].alive = false;
        self.face_slots.release(self.faces[f].slot);
    }

    /// `BM_edge_kill`: kills the faces around `e`, then `e`.
    fn edge_kill(&mut self, e: usize) {
        while self.edges[e].l != NONE {
            let f = self.loops[self.edges[e].l].f;
            self.face_kill(f);
        }
        let [v1, v2] = self.edges[e].v;
        self.disk_edge_remove(e, v1);
        self.disk_edge_remove(e, v2);
        self.edges[e].alive = false;
        self.edge_slots.release(self.edges[e].slot);
    }

    /// `BM_vert_kill`: kills the edges around `v`, then `v`.
    fn vert_kill(&mut self, v: usize) {
        while self.verts[v].edge != NONE {
            let e = self.verts[v].edge;
            self.edge_kill(e);
        }
        self.verts[v].alive = false;
    }

    /// `BM_ITER_ELEM(BM_FACES_OF_VERT)`: the face of each loop around `v`, in `BM_LOOPS_OF_VERT` order.
    fn faces_of_vert(&self, v: usize) -> Vec<usize> {
        self.loops_of_vert(v)
            .into_iter()
            .map(|l| self.loops[l].f)
            .collect()
    }

    /// `BM_LOOPS_OF_VERT` order, as `bmiter__loop_of_vert_begin` and `_step`.
    fn loops_of_vert(&self, v: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut count = self.disk_facevert_count(v);
        if count == 0 {
            return out;
        }
        let mut l_first = self.disk_faceloop_find_first(self.verts[v].edge, v);
        let mut e_next = self.loops[l_first].e;
        let mut l_next = l_first;
        while count > 0 {
            let l_curr = l_next;
            count -= 1;
            l_next = self.radial_faceloop_find_next(l_next, v);
            if l_next == l_first {
                e_next = self.disk_faceedge_find_next(e_next, v);
                l_first = self.radial_faceloop_find_first(self.edges[e_next].l, v);
                l_next = l_first;
            }
            out.push(l_curr);
        }
        out
    }

    fn radial_facevert_count(&self, l: usize, v: usize) -> usize {
        let mut count = 0;
        let mut it = l;
        loop {
            count += usize::from(self.loops[it].v == v);
            it = self.loops[it].radial_next;
            if it == l {
                return count;
            }
        }
    }

    /// `bmesh_disk_facevert_count`.
    fn disk_facevert_count(&self, v: usize) -> usize {
        let first = self.verts[v].edge;
        if first == NONE {
            return 0;
        }
        let mut count = 0;
        let mut e = first;
        loop {
            if self.edges[e].l != NONE {
                count += self.radial_facevert_count(self.edges[e].l, v);
            }
            e = self.disk_next(e, v);
            if e == first {
                return count;
            }
        }
    }

    fn disk_faceloop_find_first(&self, e: usize, v: usize) -> usize {
        let mut it = e;
        loop {
            let l = self.edges[it].l;
            if l != NONE {
                return if self.loops[l].v == v {
                    l
                } else {
                    self.loops[l].next
                };
            }
            it = self.disk_next(it, v);
            if it == e {
                return NONE;
            }
        }
    }

    fn radial_faceloop_find_first(&self, l: usize, v: usize) -> usize {
        let mut it = l;
        loop {
            if self.loops[it].v == v {
                return it;
            }
            it = self.loops[it].radial_next;
            if it == l {
                return NONE;
            }
        }
    }

    fn radial_faceloop_find_next(&self, l: usize, v: usize) -> usize {
        let mut it = self.loops[l].radial_next;
        loop {
            if self.loops[it].v == v {
                return it;
            }
            it = self.loops[it].radial_next;
            if it == l {
                return l;
            }
        }
    }

    fn disk_faceedge_find_next(&self, e: usize, v: usize) -> usize {
        let mut found = self.disk_next(e, v);
        loop {
            if self.edges[found].l != NONE && self.radial_facevert_count(self.edges[found].l, v) > 0
            {
                return found;
            }
            found = self.disk_next(found, v);
            if found == e {
                return e;
            }
        }
    }

    /// `BM_edge_other_loop`.
    fn edge_other_loop(&self, e: usize, l: usize) -> usize {
        let start = if self.loops[l].e == e {
            l
        } else {
            self.loops[l].prev
        };
        let other = self.loops[start].radial_next;
        if self.loops[other].v == self.loops[l].v {
            other
        } else {
            self.loops[other].next
        }
    }

    /// `BM_vert_step_fan_loop`.
    fn step_fan_loop(&self, l: usize, e_step: &mut usize) -> Option<usize> {
        let e_prev = *e_step;
        let e_next = if self.loops[l].e == e_prev {
            self.loops[self.loops[l].prev].e
        } else if self.loops[self.loops[l].prev].e == e_prev {
            self.loops[l].e
        } else {
            return None;
        };
        if self.is_manifold(e_next) {
            *e_step = e_next;
            Some(self.edge_other_loop(e_next, l))
        } else {
            None
        }
    }

    /// `BM_vert_is_manifold`.
    fn vert_is_manifold(&self, v: usize) -> bool {
        let first = self.verts[v].edge;
        if first == NONE {
            return false;
        }
        let mut loop_num = 0;
        let mut boundary_num = 0;
        let mut l_first = self.edges[first].l;
        let mut e_iter = first;
        loop {
            let l = self.edges[e_iter].l;
            if l == NONE || l != self.loops[self.loops[l].radial_next].radial_next {
                return false;
            }
            if self.loops[l].v == v {
                loop_num += 1;
            }
            if self.is_boundary(e_iter) {
                l_first = l;
                boundary_num += 1;
                if boundary_num == 3 {
                    return false;
                }
            } else if self.loops[self.loops[l].radial_next].v == v {
                loop_num += 1;
            }
            e_iter = self.disk_next(e_iter, v);
            if e_iter == first {
                break;
            }
        }
        let mut e_prev = self.loops[l_first].e;
        let start = if self.loops[l_first].v == v {
            l_first
        } else {
            self.loops[l_first].next
        };
        let mut l_iter = start;
        let mut loop_num_region = 0;
        loop {
            loop_num_region += 1;
            match self.step_fan_loop(l_iter, &mut e_prev) {
                Some(next) if next != start => l_iter = next,
                _ => break,
            }
        }
        loop_num == loop_num_region
    }

    /// `BM_vert_edge_pair`.
    fn vert_edge_pair(&self, v: usize) -> Option<(usize, usize)> {
        let a = self.verts[v].edge;
        if a == NONE {
            return None;
        }
        let b = self.disk_next(a, v);
        (b != a && self.disk_next(b, v) == a).then_some((a, b))
    }

    fn vert_is_edge_pair(&self, v: usize) -> bool {
        self.vert_edge_pair(v).is_some()
    }

    /// `BM_face_vert_share_loop`.
    fn face_vert_share_loop(&self, f: usize, v: usize) -> Option<usize> {
        self.face_loops(f)
            .into_iter()
            .find(|&l| self.loops[l].v == v)
    }

    /// `BM_vert_calc_edge_angle`.
    fn vert_calc_edge_angle(&self, v: usize) -> f32 {
        let e1 = self.verts[v].edge;
        if e1 != NONE {
            let e2 = self.disk_next(e1, v);
            if e1 != e2 && e1 == self.disk_next(e2, v) {
                let v1 = self.other_vert(e1, v);
                let v2 = self.other_vert(e2, v);
                return std::f32::consts::PI
                    - angle_v3v3v3(self.verts[v1].co, self.verts[v].co, self.verts[v2].co);
            }
        }
        UNIT_TO_ANGLE
    }

    /// `BM_edge_calc_face_angle`.
    fn edge_calc_face_angle(&self, e: usize) -> f32 {
        if self.is_manifold(e) {
            let l1 = self.edges[e].l;
            let l2 = self.loops[l1].radial_next;
            angle_normalized(
                self.faces[self.loops[l1].f].no,
                self.faces[self.loops[l2].f].no,
            )
        } else {
            UNIT_TO_ANGLE
        }
    }

    /// `BM_face_find_double`: an existing face with the same loops, in either winding.
    fn face_find_double(&self, f: usize) -> Option<usize> {
        let l_first = self.faces[f].first;
        let len = self.faces[f].len;
        let mut l_iter = self.loops[l_first].radial_next;
        while l_iter != l_first {
            let other = self.loops[l_iter].f;
            if self.faces[other].len == len {
                let same_vertex = self.loops[l_iter].v == self.loops[l_first].v;
                let mut a = l_first;
                let mut b = l_iter;
                let mut same = true;
                loop {
                    if self.loops[a].e != self.loops[b].e {
                        same = false;
                        break;
                    }
                    a = if same_vertex {
                        self.loops[a].next
                    } else {
                        self.loops[a].prev
                    };
                    b = self.loops[b].next;
                    if b == l_iter {
                        break;
                    }
                }
                if same {
                    return Some(other);
                }
            }
            l_iter = self.loops[l_iter].radial_next;
        }
        None
    }
}

impl BMesh {
    /// `bm_loop_systag_count_radial`: the joined faces around `l`'s edge, `l`'s own face included.
    fn radial_jf_count(&self, l: usize) -> usize {
        let mut count = 0;
        let mut it = l;
        loop {
            count += usize::from(self.faces[self.loops[it].f].jf);
            it = self.loops[it].radial_next;
            if it == l {
                return count;
            }
        }
    }

    /// `bm_vert_is_manifold_flagged`: every face around `v` is joined and no edge at `v` is a boundary.
    fn verts_all_jf(&self, v: usize) -> bool {
        let first = self.verts[v].edge;
        if first == NONE {
            return false;
        }
        let mut e = first;
        loop {
            let l = self.edges[e].l;
            if l == NONE || self.is_boundary(e) {
                return false;
            }
            let mut it = l;
            loop {
                if !self.faces[self.loops[it].f].jf {
                    return false;
                }
                it = self.loops[it].radial_next;
                if it == l {
                    break;
                }
            }
            e = self.disk_next(e, v);
            if e == first {
                return true;
            }
        }
    }

    fn clear_jf_faces(&mut self, faces: &[usize]) {
        for &f in faces {
            self.faces[f].jf = false;
        }
    }

    /// `bm_edges_sort_winding`: the ring of `edges` from `v1` to `v2`, as vertices and edges in order,
    /// written to `verts` and `sorted`. Membership of the edges and of their vertices lives in the
    /// stamped marks, so each call costs time linear in `edges` and allocates nothing.
    fn sort_winding(
        &mut self,
        v1: usize,
        v2: usize,
        edges: &[usize],
        verts: &mut Vec<usize>,
        sorted: &mut Vec<usize>,
    ) -> bool {
        self.mark_stamp += 1;
        let stamp = self.mark_stamp;
        self.edge_mark.resize(self.edges.len(), 0);
        self.vert_mark.resize(self.verts.len(), 0);
        for &e in edges {
            self.edge_mark[e] = stamp;
            for v in self.edges[e].v {
                self.vert_mark[v] = stamp;
            }
        }
        verts.clear();
        sorted.clear();
        let first = self.verts[v1].edge;
        if first == NONE {
            return false;
        }
        let mut v_iter = v1;
        let mut e_iter = first;
        loop {
            if self.edge_mark[e_iter] == stamp && self.other_vert(e_iter, v_iter) == v2 {
                break;
            }
            e_iter = self.disk_next(e_iter, v_iter);
            if e_iter == first {
                return false;
            }
        }
        let mut e_first = first;
        loop {
            if self.edge_mark[e_iter] == stamp {
                if self.vert_mark[v_iter] != stamp {
                    return false;
                }
                self.edge_mark[e_iter] = 0;
                sorted.push(e_iter);
                self.vert_mark[v_iter] = 0;
                verts.push(v_iter);
                v_iter = self.other_vert(e_iter, v_iter);
                if sorted.len() == edges.len() {
                    return v_iter == verts[0];
                }
                e_first = e_iter;
            }
            e_iter = self.disk_next(e_iter, v_iter);
            if e_iter == e_first {
                return false;
            }
        }
    }

    /// `BM_face_create_ngon`. The ring is sorted into the mesh's scratch vectors, which keep their
    /// capacity between joins.
    fn face_create_ngon(
        &mut self,
        v1: usize,
        v2: usize,
        edges: &[usize],
        example: usize,
    ) -> Option<usize> {
        let mut verts = std::mem::take(&mut self.ring_verts);
        let mut sorted = std::mem::take(&mut self.ring_edges);
        let made = self
            .sort_winding(v1, v2, edges, &mut verts, &mut sorted)
            .then(|| self.face_create(&verts, &sorted, example));
        self.ring_verts = verts;
        self.ring_edges = sorted;
        made
    }

    /// `BM_faces_join`. Returns the joined face and, with `want_double`, the existing face that the
    /// joined face doubles (the joined face is kept). Without it, a double is replaced by that face.
    fn faces_join(
        &mut self,
        faces: &[usize],
        do_del: bool,
        want_double: bool,
    ) -> (Option<usize>, Option<usize>) {
        if let [only] = faces {
            return (Some(*only), None);
        }
        for &f in faces {
            self.faces[f].jf = true;
        }
        let mut edges = Vec::new();
        let mut deledges = Vec::new();
        let mut delverts = Vec::new();
        let mut ends: Option<(usize, usize)> = None;
        for &f in faces {
            // Walks the ring by index: the body edits the mesh.
            let first = self.faces[f].first;
            let mut l = first;
            loop {
                let rlen = self.radial_jf_count(l);
                if rlen > 2 {
                    // The input faces do not form a contiguous manifold region.
                    self.clear_jf_faces(faces);
                    return (None, None);
                }
                let e = self.loops[l].e;
                if rlen == 1 {
                    edges.push(e);
                    if ends.is_none() {
                        let v1 = self.loops[l].v;
                        ends = Some((v1, self.other_vert(e, v1)));
                    }
                } else if rlen == 2 {
                    let [a, b] = self.edges[e].v;
                    let d1 = self.verts_all_jf(a);
                    let d2 = self.verts_all_jf(b);
                    if !d1 && !d2 && !self.edges[e].jf {
                        // An edge that is also the side of another face is kept.
                        if !self.edge_face_count_over(e, 2) {
                            if do_del {
                                deledges.push(e);
                            }
                            self.edges[e].jf = true;
                        }
                    } else {
                        for (v, d) in [(a, d1), (b, d2)] {
                            if d && !self.verts[v].jf {
                                if do_del {
                                    delverts.push(v);
                                }
                                self.verts[v].jf = true;
                            }
                        }
                    }
                }
                l = self.loops[l].next;
                if l == first {
                    break;
                }
            }
        }
        let made = match ends {
            Some((v1, v2)) if !edges.is_empty() => self.face_create_ngon(v1, v2, &edges, faces[0]),
            _ => None,
        };
        let Some(mut f_new) = made else {
            self.clear_jf_faces(faces);
            return (None, None);
        };
        let mut double = None;
        let mut reused = false;
        if let Some(existing) = self.face_find_double(f_new) {
            if want_double {
                double = Some(existing);
            } else {
                self.face_kill(f_new);
                f_new = existing;
                reused = true;
            }
        }
        if !reused {
            // Each new loop takes the data of the joined face's loop on the same boundary edge.
            let first = self.faces[f_new].first;
            let mut l_iter = first;
            loop {
                let mut l2 = self.loops[l_iter].radial_next;
                while l2 != l_iter && !self.faces[self.loops[l2].f].jf {
                    l2 = self.loops[l2].radial_next;
                }
                if l2 != l_iter {
                    if self.loops[l2].v != self.loops[l_iter].v {
                        l2 = self.loops[l2].next;
                    }
                    self.copy_loop_uv(l_iter, l2);
                    self.loops[l_iter].src = self.loops[l2].src;
                }
                l_iter = self.loops[l_iter].next;
                if l_iter == first {
                    break;
                }
            }
        }
        self.clear_jf_faces(faces);
        if do_del {
            for e in deledges {
                self.edge_kill(e);
            }
            for v in delverts {
                self.vert_kill(v);
            }
        } else {
            for &f in faces {
                self.face_kill(f);
            }
        }
        (Some(f_new), double)
    }

    /// `BM_faces_join_pair`.
    fn faces_join_pair(
        &mut self,
        l_a: usize,
        l_b: usize,
        do_del: bool,
        want_double: bool,
    ) -> (Option<usize>, Option<usize>) {
        if self.loops[l_a].v == self.loops[l_b].v {
            let f = self.loops[l_b].f;
            self.loop_reverse(f);
        }
        let faces = [self.loops[l_a].f, self.loops[l_b].f];
        self.faces_join(&faces, do_del, want_double)
    }

    /// `bmesh_kernel_loop_reverse`: reverses the winding of face `f`.
    fn loop_reverse(&mut self, f: usize) {
        let l_first = self.faces[f].first;
        let prev_first = self.loops[l_first].prev;
        let mut e_prev = self.loops[prev_first].e;
        let mut l_prev_radial_next = self.loops[prev_first].radial_next;
        let mut l_prev_radial_prev = self.loops[prev_first].radial_prev;
        let mut is_prev_boundary = l_prev_radial_next == self.loops[l_prev_radial_next].radial_next;
        let mut l_iter = l_first;
        loop {
            let e_iter = self.loops[l_iter].e;
            let l_iter_radial_next = self.loops[l_iter].radial_next;
            let l_iter_radial_prev = self.loops[l_iter].radial_prev;
            let is_iter_boundary = l_iter_radial_next == self.loops[l_iter_radial_next].radial_next;
            if is_prev_boundary {
                self.loops[l_iter].radial_next = l_iter;
                self.loops[l_iter].radial_prev = l_iter;
            } else {
                self.loops[l_iter].radial_next = l_prev_radial_next;
                self.loops[l_iter].radial_prev = l_prev_radial_prev;
                self.loops[l_prev_radial_next].radial_prev = l_iter;
                self.loops[l_prev_radial_prev].radial_next = l_iter;
            }
            if self.edges[e_iter].l == l_iter {
                self.edges[e_iter].l = self.loops[l_iter].next;
            }
            self.loops[l_iter].e = e_prev;
            let (next, prev) = (self.loops[l_iter].next, self.loops[l_iter].prev);
            self.loops[l_iter].next = prev;
            self.loops[l_iter].prev = next;
            e_prev = e_iter;
            l_prev_radial_next = l_iter_radial_next;
            l_prev_radial_prev = l_iter_radial_prev;
            is_prev_boundary = is_iter_boundary;
            l_iter = self.loops[l_iter].prev;
            if l_iter == l_first {
                break;
            }
        }
    }

    /// `BM_face_split` with a null example edge: divides `f` along the loops `l_a` and `l_b`. Returns
    /// the new face and its loop at `l_a`'s vertex.
    fn split_face(
        &mut self,
        f: usize,
        l_a: usize,
        l_b: usize,
        no_double: bool,
    ) -> Option<(usize, usize)> {
        if self.loops[l_a].next == l_b || self.loops[l_a].prev == l_b {
            return None;
        }
        Some(self.split_face_make_edge(f, l_a, l_b, no_double))
    }

    /// `bmesh_kernel_split_face_make_edge`: `f` keeps the loops from `l_v1` up to `l_v2`, and the new
    /// face takes the loops from `l_v2` up to `l_v1`, joined by a new edge.
    fn split_face_make_edge(
        &mut self,
        f: usize,
        l_v1: usize,
        l_v2: usize,
        no_double: bool,
    ) -> (usize, usize) {
        let (v1, v2) = (self.loops[l_v1].v, self.loops[l_v2].v);
        let e = self.edge_create(v1, v2, no_double);
        let source = self.faces[f].clone();
        let f2 = self.faces.len();
        let slot = self.face_slots.alloc();
        self.faces.push(BFace {
            first: NONE,
            len: 0,
            no: [0.0; 3],
            mat: source.mat,
            origin: source.origin,
            alive: true,
            jf: false,
            slot,
        });
        let l_f1 = self.loop_create(v2, e, f, l_v2);
        let l_f2 = self.loop_create(v1, e, f2, l_v1);
        let (prev_v1, prev_v2) = (self.loops[l_v1].prev, self.loops[l_v2].prev);
        self.loops[l_f1].prev = prev_v2;
        self.loops[l_f2].prev = prev_v1;
        self.loops[prev_v2].next = l_f1;
        self.loops[prev_v1].next = l_f2;
        self.loops[l_f1].next = l_v1;
        self.loops[l_f2].next = l_v2;
        self.loops[l_v1].prev = l_f1;
        self.loops[l_v2].prev = l_f2;

        let first_f = self.faces[f].first;
        let mut first_in_f1 = false;
        let mut it = l_f1;
        loop {
            if it == first_f {
                first_in_f1 = true;
            }
            it = self.loops[it].next;
            if it == l_f1 {
                break;
            }
        }
        if first_in_f1 {
            self.faces[f2].first = if self.loops[first_f].prev == l_f1 {
                self.loops[l_f2].prev
            } else if self.loops[first_f].next == l_f1 {
                self.loops[l_f2].next
            } else {
                l_f2
            };
        } else {
            self.faces[f2].first = first_f;
            self.faces[f].first = if self.loops[first_f].prev == l_f2 {
                self.loops[l_f1].prev
            } else if self.loops[first_f].next == l_f2 {
                self.loops[l_f1].next
            } else {
                l_f1
            };
        }
        let f2_first = self.faces[f2].first;
        let mut f2_len = 0;
        let mut it = f2_first;
        loop {
            self.loops[it].f = f2;
            f2_len += 1;
            it = self.loops[it].next;
            if it == f2_first {
                break;
            }
        }
        self.faces[f2].len = f2_len;
        self.radial_append(e, l_f1);
        self.radial_append(e, l_f2);
        let f_first = self.faces[f].first;
        let mut f_len = 0;
        let mut it = f_first;
        loop {
            f_len += 1;
            it = self.loops[it].next;
            if it == f_first {
                break;
            }
        }
        self.faces[f].len = f_len;
        (f2, l_f2)
    }

    /// `bmesh_kernel_join_edge_kill_vert` as `BM_vert_collapse_edge` calls it: collapses the two-edge
    /// vertex `v_kill` into its other neighbour along `e_kill`. Returns the surviving edge.
    fn join_edge_kill_vert(&mut self, e_kill: usize, v_kill: usize) -> Option<usize> {
        if !self.vert_in_edge(e_kill, v_kill) || self.disk_count_at_most(v_kill, 3) != 2 {
            return None;
        }
        let e_old = self.disk_next(e_kill, v_kill);
        let v_target = self.other_vert(e_kill, v_kill);
        let v_old = self.other_vert(e_old, v_kill);
        if self.vert_in_edge(e_old, v_kill) && self.vert_in_edge(e_old, v_target) {
            return None;
        }
        let e_splice = self.edge_exists(v_target, v_old);
        self.disk_vert_replace(e_old, v_target, v_kill);
        self.disk_edge_remove(e_kill, v_target);
        let mut degenerate = Vec::new();
        let mut duplicate_candidates = Vec::new();
        let head = self.edges[e_kill].l;
        if head != NONE {
            let mut l_kill = head;
            loop {
                let next = self.loops[l_kill].next;
                if self.loops[next].v == v_kill {
                    self.loops[next].v = v_target;
                }
                let prev = self.loops[l_kill].prev;
                self.loops[next].prev = prev;
                self.loops[prev].next = next;
                let f = self.loops[l_kill].f;
                if self.faces[f].first == l_kill {
                    self.faces[f].first = next;
                }
                self.faces[f].len = self.faces[f].len.saturating_sub(1);
                if self.faces[f].len < 3 {
                    degenerate.push(f);
                } else {
                    duplicate_candidates.push(f);
                }
                l_kill = self.loops[l_kill].radial_next;
                if l_kill == head {
                    break;
                }
            }
        }
        self.edges[e_kill].alive = false;
        self.edge_slots.release(self.edges[e_kill].slot);
        self.verts[v_kill].alive = false;
        if let Some(e_splice) = e_splice {
            self.edge_splice(e_old, e_splice);
        }
        while let Some(f) = degenerate.pop() {
            if self.faces[f].alive {
                self.face_kill(f);
            }
        }
        while let Some(f) = duplicate_candidates.pop() {
            if self.faces[f].alive && self.face_find_double(f).is_some() {
                self.face_kill(f);
            }
        }
        Some(e_old)
    }

    /// `BM_edge_splice`: moves the faces of `src` onto `dst`, then kills `src`.
    fn edge_splice(&mut self, dst: usize, src: usize) {
        while self.edges[src].l != NONE {
            let l = self.edges[src].l;
            self.radial_remove(src, l);
            self.radial_append(dst, l);
        }
        self.edge_kill(src);
    }

    /// `BM_vert_collapse_faces` with `do_del`, `kill_degenerate_faces` and `kill_duplicate_faces` set.
    /// With `join_faces` the faces around `v_kill` are joined and the edge between its neighbours is
    /// split back in.
    fn vert_collapse_faces(
        &mut self,
        e_kill: usize,
        v_kill: usize,
        join_faces: bool,
    ) -> Option<usize> {
        if !join_faces {
            return self.join_edge_kill_vert(e_kill, v_kill);
        }
        let tv = self.other_vert(e_kill, v_kill);
        let e2 = self.disk_next(e_kill, v_kill);
        let tv2 = self.other_vert(e2, v_kill);
        let faces = self.faces_of_vert(v_kill);
        if faces.len() < 2 {
            return None;
        }
        let (joined, double) = self.faces_join(&faces, true, true);
        let f2 = joined?;
        if let Some(double) = double
            && self.faces[double].alive
        {
            self.face_kill(double);
        }
        let la = self.face_vert_share_loop(f2, tv)?;
        let lb = self.face_vert_share_loop(f2, tv2)?;
        let (_, l_new) = self.split_face(f2, la, lb, false)?;
        Some(self.loops[l_new].e)
    }

    /// `BM_vert_collapse_edge`.
    fn vert_collapse_edge(&mut self, e_kill: usize, v_kill: usize) -> Option<usize> {
        self.vert_collapse_faces(e_kill, v_kill, false)
    }

    /// `BM_edge_share_face_check`.
    fn edge_share_face(&self, e1: usize, e2: usize) -> bool {
        self.edges[e1].l != NONE
            && self.edges[e2].l != NONE
            && self
                .radial_loops(e1)
                .into_iter()
                .any(|l| self.edge_in_face(e2, self.loops[l].f))
    }

    /// `BM_edge_in_face`.
    fn edge_in_face(&self, e: usize, f: usize) -> bool {
        self.radial_loops(e)
            .into_iter()
            .any(|l| self.loops[l].f == f)
    }

    /// `BM_disk_dissolve`: dissolves a manifold vertex of three or more edges.
    fn disk_dissolve(&mut self, v: usize) -> bool {
        if !self.vert_is_manifold(v) {
            return false;
        }
        let base = self.verts[v].edge;
        let mut keepedge = NONE;
        let mut baseedge = NONE;
        let mut len = 0;
        if base != NONE {
            let mut e = base;
            loop {
                e = self.disk_next(e, v);
                if !self.edge_share_face(e, base) {
                    keepedge = e;
                    baseedge = base;
                    break;
                }
                len += 1;
                if e == base {
                    break;
                }
            }
        }
        if keepedge == NONE && len == 3 {
            let l = self.edges[base].l;
            let rn = self.loops[l].radial_next;
            if self.faces_join_pair(l, rn, true, true).0.is_none() {
                return false;
            }
            let ve = self.verts[v].edge;
            return self.vert_collapse_faces(ve, v, false).is_some();
        }
        if keepedge == NONE && len == 2 {
            let ve = self.verts[v].edge;
            let Some(e) = self.vert_collapse_faces(ve, v, true) else {
                return false;
            };
            let l = self.edges[e].l;
            if l != NONE {
                let rn = self.loops[l].radial_next;
                if l != rn && self.faces_join_pair(l, rn, true, true).0.is_none() {
                    return false;
                }
            }
            return true;
        }
        if keepedge != NONE {
            loop {
                let mut done = true;
                let mut e = self.verts[v].edge;
                loop {
                    if self.is_manifold(e) && e != baseedge && e != keepedge {
                        let l = self.edges[e].l;
                        let rn = self.loops[l].radial_next;
                        if self.faces_join_pair(l, rn, true, true).0.is_none() {
                            return false;
                        }
                        done = false;
                        break;
                    }
                    e = self.disk_next(e, v);
                    if e == self.verts[v].edge {
                        break;
                    }
                }
                if done {
                    break;
                }
            }
            let join = !self.is_boundary(baseedge);
            let Some(e) = self.vert_collapse_faces(baseedge, v, join) else {
                return false;
            };
            let l = self.edges[e].l;
            if l != NONE {
                let rn = self.loops[l].radial_next;
                if l != rn && self.faces_join_pair(l, rn, true, true).0.is_none() {
                    return false;
                }
            }
        }
        true
    }

    /// `BM_vert_dissolve`.
    fn vert_dissolve(&mut self, v: usize) -> bool {
        let len = self.disk_count_at_most(v, 3);
        if len == 1 {
            self.vert_kill(v);
            return true;
        }
        let e = self.verts[v].edge;
        if !self.vert_is_manifold(v) {
            if e == NONE {
                self.vert_kill(v);
                return true;
            }
            if self.edges[e].l == NONE && len == 2 {
                return self.vert_collapse_edge(e, v).is_some();
            }
            return false;
        }
        if len == 2 && self.disk_facevert_count(v) == 1 {
            return self.vert_collapse_edge(e, v).is_some();
        }
        self.disk_dissolve(v)
    }

    /// `overlap_walk`: `BM_face_pair_overlap_check_subset_same_winding` (`swap` false) and
    /// `..._swap_winding` (`swap` true).
    fn overlap_walk(&self, mut a: usize, a_end: usize, mut b: usize, swap: bool) -> bool {
        loop {
            if self.loops[b].v != self.loops[a].v {
                return false;
            }
            if a == a_end {
                return true;
            }
            a = self.loops[a].next;
            b = if swap {
                self.loops[b].prev
            } else {
                self.loops[b].next
            };
        }
    }

    /// `BM_face_exists_subset_from_face`: an existing face that holds the `l_a`..`l_b` side of `f_len`.
    fn face_exists_subset(&self, l_a: usize, l_b: usize, f_len: usize) -> Option<usize> {
        let e_exists = self.edge_exists(self.loops[l_a].v, self.loops[l_b].v)?;
        let head = self.edges[e_exists].l;
        if head == NONE {
            return None;
        }
        let mut it = head;
        loop {
            let f = self.loops[it].f;
            if self.faces[f].len == f_len {
                let swap = self.loops[it].v == self.loops[l_a].v;
                let beg = if swap { it } else { self.loops[it].next };
                let overlap = if swap {
                    self.overlap_walk(
                        self.loops[l_a].next,
                        self.loops[l_b].prev,
                        self.loops[beg].prev,
                        true,
                    )
                } else {
                    self.overlap_walk(
                        self.loops[l_a].next,
                        self.loops[l_b].prev,
                        self.loops[beg].next,
                        false,
                    )
                };
                if overlap {
                    return Some(f);
                }
            }
            it = self.loops[it].radial_next;
            if it == head {
                return None;
            }
        }
    }

    /// `BM_face_split_check_double_face`.
    fn face_split_check_double(&self, l_a: usize, l_b: usize, f_len: usize) -> bool {
        let side_b_len = self.faces[self.loops[l_a].f].len + 2 - f_len;
        self.face_exists_subset(l_a, l_b, f_len).is_some()
            || self.face_exists_subset(l_b, l_a, side_b_len).is_some()
    }

    /// `BM_face_exists_multi_edge`: whether a face is bounded by exactly the closed ring `earr`.
    fn face_exists_multi_edge(&self, earr: &[usize]) -> bool {
        let mut varr: Vec<usize> = Vec::with_capacity(earr.len());
        for &e in earr {
            for v in self.edges[e].v {
                if !varr.contains(&v) {
                    varr.push(v);
                }
            }
        }
        if varr.len() != earr.len() {
            return false;
        }
        self.face_exists_multi(&varr, earr)
    }

    /// `BM_face_exists_multi`.
    fn face_exists_multi(&self, varr: &[usize], earr: &[usize]) -> bool {
        let verts: HashSet<usize> = varr.iter().copied().collect();
        let edges: HashSet<usize> = earr.iter().copied().collect();
        let mut faces: HashSet<usize> = HashSet::new();
        let mut tot_tag = 0;
        for &e in earr {
            for f in self.edge_faces(e) {
                if !faces.contains(&f)
                    && self
                        .face_loops(f)
                        .into_iter()
                        .all(|l| verts.contains(&self.loops[l].v))
                {
                    faces.insert(f);
                    tot_tag += 1;
                }
            }
        }
        if tot_tag == 0 {
            return false;
        }
        for &v in varr {
            for e in self.disk_edges(v) {
                let [a, b] = self.edges[e].v;
                if !edges.contains(&e) && verts.contains(&a) && verts.contains(&b) {
                    let tagged = self
                        .edge_faces(e)
                        .into_iter()
                        .filter(|f| faces.contains(f))
                        .count();
                    if tagged != 2 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// The faces of `e`, one per radial loop.
    fn edge_faces(&self, e: usize) -> Vec<usize> {
        self.radial_loops(e)
            .into_iter()
            .map(|l| self.loops[l].f)
            .collect()
    }

    /// `BM_face_calc_normal`. Rings of more than four corners are summed around the ring without
    /// collecting their positions, in the order [`face_normal`] sums them.
    fn calc_face_normal(&self, f: usize) -> [f32; 3] {
        let position = |l: usize| self.verts[self.loops[l].v].co;
        if self.faces[f].len <= 4 {
            let cos: Vec<[f32; 3]> = self.face_loop_iter(f).map(position).collect();
            return face_normal(&cos);
        }
        let mut normal = [0.0_f32; 3];
        let mut previous = position(self.loops[self.faces[f].first].prev);
        for l in self.face_loop_iter(f) {
            let current = position(l);
            newell_step(&mut normal, previous, current);
            previous = current;
        }
        normalize(normal)
    }

    /// `BM_face_normal_update`.
    fn face_normal_update(&mut self, f: usize) {
        self.faces[f].no = self.calc_face_normal(f);
    }
}

impl BMesh {
    /// `bm_vert_loops_from_vert_edges`: the loops at `v` around its edges, in edge order.
    fn loops_from_vert_edges(&self, v: usize, v_edges: &[usize]) -> Vec<usize> {
        let mut out = Vec::new();
        for &e in v_edges {
            if self.edges[e].l == NONE {
                continue;
            }
            for l in self.radial_loops(e) {
                if self.loops[l].v == v {
                    if out.len() == VERT_DISSOLVE_MAX {
                        return out;
                    }
                    out.push(l);
                }
            }
        }
        out
    }

    /// `bm_vert_dissolve_fan_makes_double_whole_face`.
    fn fan_makes_double_whole_face(&self, v_loops: &[usize]) -> bool {
        let mut ears: Vec<usize> = Vec::with_capacity(v_loops.len());
        for &l in v_loops {
            let e_opposite = if self.faces[self.loops[l].f].len > 3 {
                let prev_v = self.loops[self.loops[l].prev].v;
                let next_v = self.loops[self.loops[l].next].v;
                let Some(e) = self.edge_exists(prev_v, next_v) else {
                    return false;
                };
                if ears.len() == VERT_DISSOLVE_MAX {
                    return false;
                }
                e
            } else {
                self.loops[self.loops[l].next].e
            };
            ears.push(e_opposite);
        }
        ears.len() >= 3 && self.face_exists_multi_edge(&ears)
    }

    /// `bm_vert_dissolve_fan_makes_double_with_existing`.
    fn fan_makes_double_with_existing(&self, v_loops: &[usize]) -> bool {
        v_loops.iter().any(|&l| {
            self.faces[self.loops[l].f].len > 3
                && self.face_split_check_double(self.loops[l].prev, self.loops[l].next, 3)
        })
    }

    /// `verts_canonical_collapsed_fn`: the ring left when the corner's vertex collapses out, rotated to
    /// start at its smallest vertex and oriented by the smaller of that vertex's neighbours. Vertex
    /// indices stand in for Blender's vertex pointers, which order the same way for these meshes.
    fn collapsed_ring(&self, l_vert: usize) -> Vec<usize> {
        let verts_num = self.faces[self.loops[l_vert].f].len - 1;
        let mut verts = Vec::with_capacity(verts_num);
        let mut l = self.loops[l_vert].next;
        while l != l_vert {
            verts.push(self.loops[l].v);
            l = self.loops[l].next;
        }
        let mut i_min = 0;
        for i in 1..verts_num {
            if verts[i] < verts[i_min] {
                i_min = i;
            }
        }
        let i_next = if i_min + 1 == verts_num { 0 } else { i_min + 1 };
        let i_prev = if i_min == 0 { verts_num - 1 } else { i_min - 1 };
        let step = if verts[i_next] < verts[i_prev] {
            1
        } else {
            verts_num - 1
        };
        let mut ordered = Vec::with_capacity(verts_num);
        let mut j = i_min;
        for _ in 0..verts_num {
            ordered.push(verts[j]);
            j += step;
            if j >= verts_num {
                j -= verts_num;
            }
        }
        ordered
    }

    /// `bm_vert_dissolve_fan_makes_double_with_self`.
    fn fan_makes_double_with_self(&self, v_loops: &[usize]) -> bool {
        let mut loops: Vec<usize> = v_loops
            .iter()
            .copied()
            .filter(|&l| self.faces[self.loops[l].f].len > 3)
            .collect();
        if loops.len() < 2 {
            return false;
        }
        loops.sort_by_key(|&l| self.faces[self.loops[l].f].len);
        let mut start = 0;
        while start < loops.len() {
            let group_len = self.faces[self.loops[loops[start]].f].len;
            let mut end = start + 1;
            while end < loops.len() && self.faces[self.loops[loops[end]].f].len == group_len {
                end += 1;
            }
            if end - start >= 2 {
                let collapsed: Vec<Vec<usize>> = loops[start..end]
                    .iter()
                    .map(|&l| self.collapsed_ring(l))
                    .collect();
                for (i, ring) in collapsed.iter().enumerate() {
                    if collapsed[i + 1..].contains(ring) {
                        return true;
                    }
                }
            }
            start = end;
        }
        false
    }

    /// `bm_vert_dissolve_fan_or_chain_test`: the dissolve method of `v`, with the loops around it for
    /// fans. `check_for_duplicates` also rejects dissolves that would create a duplicate face.
    fn dissolve_method(
        &self,
        v: usize,
        check_for_duplicates: bool,
    ) -> Option<(DissolveMethod, Vec<usize>)> {
        let v_edges = self.disk_edges(v);
        if v_edges.len() > VERT_DISSOLVE_MAX {
            return None;
        }
        let mut boundary = Vec::new();
        let mut tot_boundary = 0;
        let mut tot_manifold = 0;
        let mut tot_wire = 0;
        for &e in &v_edges {
            if self.is_boundary(e) {
                if boundary.len() < 2 {
                    boundary.push(e);
                }
                tot_boundary += 1;
            } else if self.is_manifold(e) {
                tot_manifold += 1;
            } else if self.is_wire(e) {
                tot_wire += 1;
            } else {
                return None;
            }
        }
        let tot_edge = v_edges.len();
        if (tot_edge == 4 && tot_boundary == 0 && tot_manifold == 4)
            || (tot_edge == 3 && tot_boundary == 0 && tot_manifold == 3)
        {
            let mut v_loops = Vec::new();
            if check_for_duplicates {
                v_loops = self.loops_from_vert_edges(v, &v_edges);
                if self.fan_makes_double_whole_face(&v_loops)
                    || self.fan_makes_double_with_existing(&v_loops)
                    || self.fan_makes_double_with_self(&v_loops)
                {
                    return None;
                }
            }
            return Some((DissolveMethod::InteriorFan, v_loops));
        }
        if tot_edge == 3 && tot_boundary == 2 && tot_manifold == 1 {
            let mut v_loops = Vec::new();
            if check_for_duplicates {
                let a = self.other_vert(boundary[0], v);
                let b = self.other_vert(boundary[1], v);
                // Joining across an existing face would remove vertices other than `v`.
                if let Some(e_span) = self.edge_exists(a, b)
                    && self.edges[e_span].l != NONE
                {
                    return None;
                }
                v_loops = self.loops_from_vert_edges(v, &v_edges);
                if self.fan_makes_double_with_existing(&v_loops)
                    || self.fan_makes_double_with_self(&v_loops)
                {
                    return None;
                }
            }
            return Some((DissolveMethod::BoundaryFan, v_loops));
        }
        if tot_edge == 2 && tot_wire == 2 {
            return Some((DissolveMethod::WireChain, Vec::new()));
        }
        if tot_edge == 2 && tot_manifold == 2 {
            return Some((DissolveMethod::InteriorChain, Vec::new()));
        }
        None
    }

    /// `bm_vert_dissolve_fan_or_chain`: dissolves `v` alone, re-testing its topology first.
    fn dissolve_fan_or_chain(&mut self, v: usize) -> bool {
        let Some((method, v_loops)) = self.dissolve_method(v, true) else {
            return false;
        };
        match method {
            DissolveMethod::WireChain | DissolveMethod::InteriorChain => {
                let e = self.verts[v].edge;
                self.vert_collapse_edge(e, v).is_some()
            }
            DissolveMethod::BoundaryFan | DissolveMethod::InteriorFan => {
                for l in v_loops {
                    if self.faces[self.loops[l].f].len > 3 {
                        let f = self.loops[l].f;
                        let (prev, next) = (self.loops[l].prev, self.loops[l].next);
                        self.split_face(f, prev, next, true);
                    }
                }
                self.vert_dissolve(v)
            }
        }
    }

    /// `bm_tag_untagged_neighbors`: tags the untagged neighbours of `starts` and returns them.
    fn tag_neighbors(&self, starts: &[usize], desired: i8, tags: &mut [i8]) -> Vec<usize> {
        let mut tagged = Vec::new();
        for &v in starts {
            for e in self.disk_edges(v) {
                let other = self.other_vert(e, v);
                if tags[other] == TAG_INIT {
                    tags[other] = desired;
                    tagged.push(other);
                }
            }
        }
        tagged
    }

    /// `BM_mesh_decimate_unsubdivide_ex`: each iteration tags the dissolvable vertices in alternating
    /// regions and dissolves one set, which turns a grid into one of half its density.
    fn unsubdivide(&mut self, iterations: usize) {
        for _ in 0..iterations {
            let mut tags = vec![TAG_IGNORE; self.verts.len()];
            for (v, tag) in tags.iter_mut().enumerate() {
                if self.verts[v].alive && self.dissolve_method(v, false).is_some() {
                    *tag = TAG_INIT;
                }
            }
            let mut marked = false;
            for start in 0..self.verts.len() {
                if !self.verts[start].alive || tags[start] != TAG_INIT {
                    continue;
                }
                tags[start] = TAG_IGNORE;
                let mut ignore = vec![start];
                loop {
                    let collapse = self.tag_neighbors(&ignore, TAG_DO_COLLAPSE, &mut tags);
                    if collapse.is_empty() {
                        break;
                    }
                    marked = true;
                    let next = self.tag_neighbors(&collapse, TAG_IGNORE, &mut tags);
                    if next.is_empty() {
                        break;
                    }
                    ignore = next;
                }
            }
            if !marked {
                break;
            }
            for (v, &tag) in tags.iter().enumerate() {
                if self.verts[v].alive && tag == TAG_DO_COLLAPSE {
                    self.dissolve_fan_or_chain(v);
                }
            }
        }
    }

    /// `bm_edge_is_delimiter`. The caller ensures `e` is manifold.
    fn edge_is_delimiter(&self, e: usize, delim: Delimit) -> bool {
        let edge = self.edges[e];
        if delim.seam && edge.seam {
            return true;
        }
        if delim.sharp && !edge.smooth {
            return true;
        }
        let l = edge.l;
        let other = self.loops[l].radial_next;
        if delim.material && self.faces[self.loops[l].f].mat != self.faces[self.loops[other].f].mat
        {
            return true;
        }
        if delim.normal && !self.is_contiguous(e) {
            return true;
        }
        delim.uv && (0..self.uv_layers).any(|layer| !self.is_contiguous_uv(e, layer))
    }

    /// `bm_vert_is_delimiter`.
    fn vert_is_delimiter(&self, v: usize, delim: Delimit) -> bool {
        if !(delim.seam || delim.sharp || delim.material || delim.normal || delim.uv) {
            return false;
        }
        self.disk_edges(v)
            .into_iter()
            .any(|e| self.is_manifold(e) && self.edge_is_delimiter(e, delim))
    }

    /// `bm_edge_calc_dissolve_error`: the cosine of the angle between the two faces, negated when the
    /// edge is contiguous, or `COST_INVALID`.
    fn edge_calc_dissolve_error(&self, e: usize, delim: Delimit) -> f32 {
        if self.is_manifold(e) && !self.edge_is_delimiter(e, delim) {
            let l = self.edges[e].l;
            let other = self.loops[l].radial_next;
            let mut cos_neg = dot(
                self.faces[self.loops[l].f].no,
                self.faces[self.loops[other].f].no,
            );
            if self.is_contiguous(e) {
                cos_neg *= -1.0;
            }
            return cos_neg;
        }
        COST_INVALID
    }

    /// `bm_vert_edge_face_angle`: the vertex's edge angle, scaled by its manifold edge's face angle.
    fn vert_edge_face_angle(&self, v: usize, delim: Delimit) -> f32 {
        let angle = self.vert_calc_edge_angle(v);
        let e = self.verts[v].edge;
        if e != NONE && self.is_manifold(e) && !self.vert_is_delimiter(v, delim) {
            return ((angle * ANGLE_TO_UNIT) * (self.edge_calc_face_angle(e) * ANGLE_TO_UNIT))
                * UNIT_TO_ANGLE;
        }
        angle
    }

    /// `bm_loop_collapse_is_degenerate`: whether collapsing the ear at `l_ear` would flip a corner or
    /// leave a vertex inside the new triangle. Coordinates are projected onto the face plane around
    /// the ear's vertex.
    fn loop_collapse_is_degenerate(&self, l_ear: usize) -> bool {
        let center = self.verts[self.loops[l_ear].v].co;
        let (n1, n2) = ortho_basis(self.faces[self.loops[l_ear].f].no);
        let project = |vert: usize| {
            let co = sub(self.verts[vert].co, center);
            [dot(n1, co), dot(n2, co)]
        };
        let prev = self.loops[l_ear].prev;
        let next = self.loops[l_ear].next;
        let prev_v = self.loops[prev].v;
        let next_v = self.loops[next].v;
        let mut tri = [project(prev_v), [0.0, 0.0], project(next_v)];
        if !self.vert_is_edge_pair(prev_v) {
            let adjacent = project(self.loops[self.loops[prev].prev].v);
            if signum_i(cross_tri(adjacent, tri[0], tri[1]))
                != signum_i(cross_tri(adjacent, tri[0], tri[2]))
            {
                return true;
            }
        }
        if !self.vert_is_edge_pair(next_v) {
            let adjacent = project(self.loops[self.loops[next].next].v);
            if signum_i(cross_tri(adjacent, tri[2], tri[1]))
                != signum_i(cross_tri(adjacent, tri[2], tri[0]))
            {
                return true;
            }
        }
        // The triangle may be concave: flip it so the inside test can assume clockwise order.
        if cross_tri(tri[0], tri[1], tri[2]) < 0.0 {
            tri.swap(1, 2);
        }
        let mut l_iter = self.loops[next].next;
        loop {
            if point_in_tri_cw(project(self.loops[l_iter].v), tri) {
                return true;
            }
            l_iter = self.loops[l_iter].next;
            if l_iter == prev {
                return false;
            }
        }
    }

    /// `bm_vert_collapse_is_degenerate`.
    fn vert_collapse_is_degenerate(&self, v: usize) -> bool {
        let Some((e0, e1)) = self.vert_edge_pair(v) else {
            return true;
        };
        if self.is_wire(e0) || self.is_wire(e1) {
            return false;
        }
        let v0 = self.other_vert(e0, v);
        let v1 = self.other_vert(e1, v);
        if cos_v3v3v3(self.verts[v0].co, self.verts[v].co, self.verts[v1].co).abs()
            < 1.0 - f32::EPSILON
        {
            let first = self.edges[e1].l;
            let mut l_iter = first;
            loop {
                if self.faces[self.loops[l_iter].f].len > 3 {
                    let l_pivot = if self.loops[l_iter].v == v {
                        l_iter
                    } else {
                        self.loops[l_iter].next
                    };
                    if self.loop_collapse_is_degenerate(l_pivot) {
                        return true;
                    }
                }
                l_iter = self.loops[l_iter].radial_next;
                if l_iter == first {
                    break;
                }
            }
        }
        false
    }

    /// `BM_mesh_decimate_dissolve_ex`: first joins the faces across the flattest edges, then collapses
    /// the vertices whose angle is below `angle_limit`.
    fn dissolve(&mut self, angle_limit: f32, do_dissolve_boundaries: bool, delim: Delimit) {
        let angle_limit_cos_neg = -angle_limit.cos();
        let einput: Vec<usize> = (0..self.edges.len())
            .filter(|&e| self.edges[e].alive)
            .collect();
        for &e in &einput {
            self.edges[e].tag = self.is_wire(e);
        }
        let mut eheap = DecimateHeap::new(self.edges.len());
        for &e in &einput {
            let cost = self.edge_calc_dissolve_error(e, delim);
            eheap.insert(e, cost);
        }
        while let Some(top) = eheap.top() {
            if eheap.value(top) >= angle_limit_cos_neg {
                break;
            }
            let mut joined = None;
            if self.is_manifold(top) {
                let l = self.edges[top].l;
                let other = self.loops[l].radial_next;
                joined = self.faces_join_pair(l, other, false, false).0;
            }
            match joined {
                Some(f) => {
                    eheap.remove(top);
                    self.face_normal_update(f);
                    for l in self.face_loop_iter(f) {
                        let j = self.loops[l].e;
                        if eheap.contains(j) {
                            let cost = self.edge_calc_dissolve_error(j, delim);
                            eheap.update(j, cost);
                        }
                    }
                }
                None => eheap.update(top, COST_INVALID),
            }
        }
        // Edges left wire by the joins are removed, with the vertices they leave loose.
        for &e in einput.iter().rev() {
            if self.edges[e].alive && self.is_wire(e) && !self.edges[e].tag {
                let [v1, v2] = self.edges[e].v;
                self.edge_kill(e);
                if self.verts[v1].edge == NONE {
                    self.vert_kill(v1);
                }
                if self.verts[v2].edge == NONE {
                    self.vert_kill(v2);
                }
            }
        }
        if do_dissolve_boundaries {
            for v in 0..self.verts.len() {
                if self.verts[v].alive && self.vert_is_edge_pair(v) {
                    let e = self.verts[v].edge;
                    self.vert_collapse_edge(e, v);
                }
            }
            return;
        }
        let mut vheap = DecimateHeap::new(self.verts.len());
        for v in 0..self.verts.len() {
            if self.verts[v].alive {
                let cost = self.vert_edge_face_angle(v, delim);
                vheap.insert(v, cost);
            }
        }
        while let Some(v) = vheap.top() {
            if vheap.value(v) >= angle_limit {
                break;
            }
            let collapsed = if self.vert_collapse_is_degenerate(v) {
                None
            } else {
                let e = self.verts[v].edge;
                self.vert_collapse_edge(e, v)
            };
            let Some(e_new) = collapsed else {
                vheap.update(v, COST_INVALID);
                continue;
            };
            vheap.remove(v);
            for l in self.radial_loops(e_new) {
                let f = self.loops[l].f;
                self.face_normal_update(f);
            }
            for vv in self.edges[e_new].v {
                if vheap.contains(vv) {
                    let cost = self.vert_edge_face_angle(vv, delim);
                    vheap.update(vv, cost);
                }
            }
            // Vertices that could not collapse before may now, around the new edge.
            for l_iter in self.radial_loops(e_new) {
                let stop = self.loops[l_iter].prev;
                let mut l_cycle = self.loops[self.loops[l_iter].next].next;
                loop {
                    let j = self.loops[l_cycle].v;
                    if vheap.contains(j) && vheap.value(j) == COST_INVALID {
                        let cost = self.vert_edge_face_angle(j, delim);
                        vheap.update(j, cost);
                    }
                    l_cycle = self.loops[l_cycle].next;
                    if l_cycle == stop {
                        break;
                    }
                }
            }
        }
    }
}

impl BMesh {
    /// `BKE_mesh_to_bmesh`: edges in mesh order, one loop per face corner in face order, with each
    /// corner's data taken from the input corner it came from.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Blender stores positions and UVs as float32"
    )]
    fn from_mesh(mesh: &Mesh, points: &[DVec3], uv_values: &[Vec<[f64; 2]>]) -> Result<Self> {
        let index_of = id_indices(mesh);
        let vertex_index = |id: &u32| {
            index_of.get(id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "modifier input references a missing vertex",
                )
            })
        };
        let flags = decimate_edge_flags(mesh);
        let mut bm = BMesh {
            verts: points
                .iter()
                .map(|point| BVert {
                    co: point.as_vec3().to_array(),
                    edge: NONE,
                    alive: true,
                    jf: false,
                })
                .collect(),
            edges: Vec::with_capacity(mesh.edges.len()),
            loops: Vec::new(),
            faces: Vec::with_capacity(mesh.faces.len()),
            free_loops: Vec::new(),
            edge_slots: PoolSlots::default(),
            face_slots: PoolSlots::default(),
            edge_mark: Vec::new(),
            vert_mark: Vec::new(),
            mark_stamp: 0,
            ring_verts: Vec::new(),
            ring_edges: Vec::new(),
            uv_layers: uv_values.len(),
        };
        let mut edge_lookup: HashMap<(usize, usize), usize> =
            HashMap::with_capacity(mesh.edges.len());
        for (index, edge) in mesh.edges.iter().enumerate() {
            let v = [
                vertex_index(&edge.vertices[0])?,
                vertex_index(&edge.vertices[1])?,
            ];
            let e = bm.edge_create(v[0], v[1], false);
            bm.edges[e].seam = flags[index].0;
            bm.edges[e].smooth = !flags[index].1;
            edge_lookup.insert(edge_key(v[0], v[1]), e);
        }
        let mut corner = 0;
        for (face_index, face) in mesh.faces.iter().enumerate() {
            let indices = face
                .vertices
                .iter()
                .map(&vertex_index)
                .collect::<Result<Vec<usize>>>()?;
            let n = indices.len();
            let mut edges = Vec::with_capacity(n);
            for k in 0..n {
                let (a, b) = (indices[k], indices[(k + 1) % n]);
                let key = edge_key(a, b);
                let e = if let Some(&e) = edge_lookup.get(&key) {
                    e
                } else {
                    let e = bm.edge_create(a, b, false);
                    edge_lookup.insert(key, e);
                    e
                };
                edges.push(e);
            }
            let first = bm.loops.len();
            let f = bm.faces.len();
            for k in 0..n {
                let uv = uv_values
                    .iter()
                    .map(|layer| {
                        let [u, v] = layer[corner + k];
                        [u as f32, v as f32]
                    })
                    .collect();
                bm.loops.push(BLoop {
                    v: indices[k],
                    e: edges[k],
                    f,
                    next: first + (k + 1) % n,
                    prev: first + (k + n - 1) % n,
                    radial_next: NONE,
                    radial_prev: NONE,
                    uv,
                    src: corner + k,
                });
                bm.radial_append(edges[k], first + k);
            }
            let cos: Vec<[f32; 3]> = indices.iter().map(|&v| bm.verts[v].co).collect();
            let slot = bm.face_slots.alloc();
            bm.faces.push(BFace {
                first,
                len: n,
                no: face_normal(&cos),
                mat: face.material_index,
                origin: Some(face_index),
                alive: true,
                jf: false,
                slot,
            });
            corner += n;
        }
        Ok(bm)
    }

    /// `BKE_mesh_from_bmesh`: the live faces with each corner's source corner, the edges that belong to
    /// no face, and every input vertex (unused ones are dropped by `decimate_compact`).
    fn to_mesh(&self, points: Vec<DVec3>, source: &Mesh) -> Result<Mesh> {
        let mut live_faces: Vec<usize> = (0..self.faces.len())
            .filter(|&f| self.faces[f].alive)
            .collect();
        live_faces.sort_by_key(|&f| self.faces[f].slot);
        let mut polygons: Vec<Polygon> = Vec::new();
        let mut face_origins = Vec::new();
        let mut corners = Vec::new();
        for f in live_faces {
            let face = &self.faces[f];
            let loops = self.face_loops(f);
            polygons.push((loops.iter().map(|&l| self.loops[l].v).collect(), face.mat));
            face_origins.push(face.origin);
            corners.extend(loops.iter().map(|&l| {
                let src = self.loops[l].src;
                if src == NONE {
                    Vec::new()
                } else {
                    vec![(src, 1.0)]
                }
            }));
        }
        let loose: Vec<[usize; 2]> = self
            .edges
            .iter()
            .filter(|edge| edge.alive && edge.l == NONE)
            .map(|edge| edge.v)
            .collect();
        let mut origin = Origin::copies((0..points.len()).map(Some), face_origins);
        origin.corners = Some(corners);
        let keep_vertices: Vec<bool> = self.verts.iter().map(|vert| vert.alive).collect();
        decimate_compact(
            points,
            polygons,
            loose,
            origin,
            source,
            &keep_vertices,
            &self.pool_edge_pairs(),
        )
    }

    /// The live edges as `BMesh` vertex-index pairs, in pool order (the index space `to_mesh` passes on).
    fn pool_edge_pairs(&self) -> Vec<[usize; 2]> {
        // `decimate_compact` remaps from the full vertex index space, dead vertices included.
        let mut live: Vec<usize> = (0..self.edges.len())
            .filter(|&e| self.edges[e].alive)
            .collect();
        live.sort_by_key(|&e| self.edges[e].slot);
        live.into_iter()
            .map(|e| {
                let [a, b] = self.edges[e].v;
                [a.min(b), a.max(b)]
            })
            .collect()
    }

    /// Output vertex of each live vertex: `to_mesh` keeps the live vertices in index order.
    fn live_vertex_index(&self) -> Vec<usize> {
        let mut next = 0;
        self.verts
            .iter()
            .map(|vert| {
                if vert.alive {
                    next += 1;
                    next - 1
                } else {
                    NONE
                }
            })
            .collect()
    }

    /// `(seam, sharp)` of each live edge keyed by its output vertex pair.
    fn output_edge_flags(&self) -> Vec<([usize; 2], (bool, bool))> {
        let output_index = self.live_vertex_index();
        self.edges
            .iter()
            .filter(|edge| edge.alive)
            .map(|edge| {
                let [a, b] = edge.v.map(|v| output_index[v]);
                ([a.min(b), a.max(b)], (edge.seam, !edge.smooth))
            })
            .collect()
    }
}

/// `BM_mesh_decimate_dissolve` and `BM_mesh_decimate_unsubdivide` on the input mesh.
pub(super) fn decimate_dissolve(
    mesh: &Mesh,
    angle_limit: f64,
    modifier: &Modifier,
) -> Result<Mesh> {
    let dissolve_boundaries = bool_param(modifier, "use_dissolve_boundaries")?;
    if angle_limit == 0.0 || mesh.faces.len() <= 3 {
        return Ok(mesh.clone());
    }
    let points = positions(mesh);
    let modes: HashSet<&str> = modifier
        .params
        .get("delimit")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let (_, uv_values) = collapse_uv_layers(mesh);
    let delim = Delimit {
        seam: modes.contains("SEAM"),
        sharp: modes.contains("SHARP"),
        material: modes.contains("MATERIAL"),
        normal: modes.contains("NORMAL"),
        uv: modes.contains("UV") && !uv_values.is_empty(),
    };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Blender stores Decimate angle limits as float32"
    )]
    let angle_limit = angle_limit as f32;
    let mut bm = BMesh::from_mesh(mesh, &points, &uv_values)?;
    bm.dissolve(angle_limit, dissolve_boundaries, delim);
    let flags = bm.output_edge_flags();
    let mut result = bm.to_mesh(points, mesh)?;
    write_dissolve_edge_flags(&mut result, flags)?;
    Ok(result)
}

pub(super) fn decimate_unsubdivide(mesh: &Mesh, iterations: usize) -> Result<Mesh> {
    if iterations == 0 || mesh.faces.len() <= 3 {
        return Ok(mesh.clone());
    }
    let points = positions(mesh);
    let mut bm = BMesh::from_mesh(mesh, &points, &[])?;
    bm.unsubdivide(iterations);
    bm.to_mesh(points, mesh)
}

/// `(seam, sharp)` of each edge of `mesh`: the compat `blender_edge_flags` OR-ed with the native `seam`
/// and `sharp` attributes, as COLLAPSE reads them. Native values are keyed `e<edge id>`.
fn decimate_edge_flags(mesh: &Mesh) -> Vec<(bool, bool)> {
    let compat = mesh
        .attributes
        .get("blender_edge_flags")
        .and_then(Value::as_array);
    let native_values = |name: &str| {
        mesh.attributes
            .get(name)
            .and_then(Value::as_object)
            .and_then(|attribute| attribute.get("values"))
            .and_then(Value::as_object)
    };
    let seam_values = native_values("seam");
    let sharp_values = native_values("sharp");
    mesh.edges
        .iter()
        .enumerate()
        .map(|(index, edge)| {
            let metadata = compat.and_then(|values| values.get(index));
            let key = format!("e{}", edge.id);
            let compat_flag = |name: &str| {
                metadata
                    .and_then(|flags| flags.get(name))
                    .and_then(Value::as_bool)
                    == Some(true)
            };
            let native_flag = |values: Option<&serde_json::Map<String, Value>>| {
                values
                    .and_then(|values| values.get(&key))
                    .and_then(Value::as_bool)
                    == Some(true)
            };
            (
                compat_flag("seam") || native_flag(seam_values),
                compat_flag("sharp") || native_flag(sharp_values),
            )
        })
        .collect()
}

/// A native edge-flag attribute and the `(seam, sharp)` value it stores.
type DissolveFlagStore = (&'static str, fn(&(bool, bool)) -> bool);

/// Writes the DISSOLVE edge flags onto the output edges the way COLLAPSE does: the native `seam` and
/// `sharp` attributes and the compat `blender_edge_flags` take the `BMesh` edge header values. `flags`
/// is keyed by output vertex pair; a pair shared by two live edges keeps the marks of both.
fn write_dissolve_edge_flags(
    mesh: &mut Mesh,
    flags: Vec<([usize; 2], (bool, bool))>,
) -> Result<()> {
    let invalid = || {
        PotError::new(
            ErrorCode::InternalError,
            "decimate edge flags do not match the rebuilt mesh",
        )
    };
    let mut by_pair: HashMap<(usize, usize), (bool, bool)> = HashMap::with_capacity(flags.len());
    for ([first, second], (seam, sharp)) in flags {
        let entry = by_pair.entry((first, second)).or_default();
        entry.0 |= seam;
        entry.1 |= sharp;
    }
    let index_of = id_indices(mesh);
    let mut output = Vec::with_capacity(mesh.edges.len());
    for edge in &mesh.edges {
        let [first, second] = edge.vertices.map(|id| index_of.get(&id).copied());
        let (Some(first), Some(second)) = (first, second) else {
            return Err(invalid());
        };
        output.push(
            by_pair
                .remove(&(first.min(second), first.max(second)))
                .ok_or_else(invalid)?,
        );
    }
    if !by_pair.is_empty() {
        return Err(invalid());
    }
    let natives: [DissolveFlagStore; 2] = [("sharp", |flags| flags.1), ("seam", |flags| flags.0)];
    for (name, flag) in natives {
        if let Some(attribute) = mesh.attributes.get_mut(name) {
            let values = attribute
                .get_mut("values")
                .and_then(Value::as_object_mut)
                .ok_or_else(invalid)?;
            values.clear();
            for (edge, flags) in mesh.edges.iter().zip(&output) {
                values.insert(format!("e{}", edge.id), json!(flag(flags)));
            }
        }
    }
    if let Some(compat) = mesh.attributes.get_mut("blender_edge_flags") {
        let values = compat.as_array_mut().ok_or_else(invalid)?;
        if values.len() != output.len() {
            return Err(invalid());
        }
        for (value, &(seam, sharp)) in values.iter_mut().zip(&output) {
            let fields = value.as_object_mut().ok_or_else(invalid)?;
            fields.insert("seam".to_owned(), json!(seam));
            fields.insert("sharp".to_owned(), json!(sharp));
        }
    }
    Ok(())
}

/// Blender's `BLI_heap` over a fixed set of nodes, where a node's id is the caller's element index.
/// Only inserted nodes take part, in insertion order. `heap_up` and `heap_down` compare with `<` as
/// Blender does, so ties resolve in the same order.
struct DecimateHeap {
    values: Vec<f32>,
    tree: Vec<usize>,
    positions: Vec<Option<usize>>,
}

impl DecimateHeap {
    fn new(nodes: usize) -> Self {
        Self {
            values: vec![0.0; nodes],
            tree: Vec::new(),
            positions: vec![None; nodes],
        }
    }

    /// `BLI_heap_top`: the minimum node.
    fn top(&self) -> Option<usize> {
        self.tree.first().copied()
    }

    fn value(&self, node: usize) -> f32 {
        self.values[node]
    }

    fn contains(&self, node: usize) -> bool {
        self.positions[node].is_some()
    }

    /// `BLI_heap_insert`.
    fn insert(&mut self, node: usize, value: f32) {
        self.values[node] = value;
        self.positions[node] = Some(self.tree.len());
        self.tree.push(node);
        self.up(self.tree.len() - 1);
    }

    fn swap(&mut self, first: usize, second: usize) {
        self.tree.swap(first, second);
        self.positions[self.tree[first]] = Some(first);
        self.positions[self.tree[second]] = Some(second);
    }

    fn less(&self, first: usize, second: usize) -> bool {
        self.values[self.tree[first]] < self.values[self.tree[second]]
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
        loop {
            let left = (index << 1) + 1;
            let right = left + 1;
            let mut smallest = index;
            if left < self.tree.len() && self.less(left, smallest) {
                smallest = left;
            }
            if right < self.tree.len() && self.less(right, smallest) {
                smallest = right;
            }
            if smallest == index {
                break;
            }
            self.swap(index, smallest);
            index = smallest;
        }
    }

    /// `BLI_heap_node_value_update`.
    fn update(&mut self, node: usize, value: f32) {
        let Some(position) = self.positions[node] else {
            return;
        };
        let old = self.values[node];
        if value < old {
            self.values[node] = value;
            self.up(position);
        } else if value > old {
            self.values[node] = value;
            self.down(position);
        }
    }

    /// `BLI_heap_remove`: moves the node to the root, then pops it.
    fn remove(&mut self, node: usize) {
        let Some(mut position) = self.positions[node] else {
            return;
        };
        while position > 0 {
            let parent = (position - 1) >> 1;
            self.swap(parent, position);
            position = parent;
        }
        let last = self.tree.len() - 1;
        if last > 0 {
            self.swap(0, last);
        }
        self.tree.pop();
        self.positions[node] = None;
        if !self.tree.is_empty() {
            self.down(0);
        }
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `normalize_v3`: the zero vector when `v` is too short to normalize.
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let d = dot(v, v);
    if d > 1.0e-35 {
        let inverse = 1.0 / d.sqrt();
        [v[0] * inverse, v[1] * inverse, v[2] * inverse]
    } else {
        [0.0; 3]
    }
}

/// One edge of Newell's method, from corner `previous` to corner `current`.
fn newell_step(normal: &mut [f32; 3], previous: [f32; 3], current: [f32; 3]) {
    normal[0] += (previous[1] - current[1]) * (previous[2] + current[2]);
    normal[1] += (previous[2] - current[2]) * (previous[0] + current[0]);
    normal[2] += (previous[0] - current[0]) * (previous[1] + current[1]);
}

/// `BM_face_calc_normal` over the corner positions of a face, in face order.
fn face_normal(cos: &[[f32; 3]]) -> [f32; 3] {
    match cos.len() {
        0..=2 => [0.0; 3],
        3 => normalize(cross(sub(cos[0], cos[1]), sub(cos[1], cos[2]))),
        4 => normalize(cross(sub(cos[0], cos[2]), sub(cos[1], cos[3]))),
        _ => {
            let mut normal = [0.0_f32; 3];
            let mut previous = cos[cos.len() - 1];
            for &current in cos {
                newell_step(&mut normal, previous, current);
                previous = current;
            }
            normalize(normal)
        }
    }
}

/// `ortho_basis_v3v3_v3`: two unit vectors orthogonal to `n` and to each other.
fn ortho_basis(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let f = n[0] * n[0] + n[1] * n[1];
    if f > f32::EPSILON {
        let d = 1.0 / f.sqrt();
        let n1 = [n[1] * d, -n[0] * d, 0.0];
        let n2 = [-n[2] * n1[1], n[2] * n1[0], n[0] * n1[1] - n[1] * n1[0]];
        (n1, n2)
    } else {
        (
            [if n[2] < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )
    }
}

/// `cross_tri_v2`.
fn cross_tri(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (a[0] - b[0]) * (b[1] - c[1]) + (a[1] - b[1]) * (c[0] - b[0])
}

/// `line_point_side_v2`.
fn line_point_side(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (a[0] - p[0]) * (b[1] - p[1]) - (b[0] - p[0]) * (a[1] - p[1])
}

/// `isect_point_tri_v2_cw`.
fn point_in_tri_cw(p: [f32; 2], tri: [[f32; 2]; 3]) -> bool {
    line_point_side(tri[0], tri[1], p) >= 0.0
        && line_point_side(tri[1], tri[2], p) >= 0.0
        && line_point_side(tri[2], tri[0], p) >= 0.0
}

/// `signum_i`.
fn signum_i(value: f32) -> i32 {
    if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    }
}

/// `saasinf`.
fn saasin(value: f32) -> f32 {
    if value <= -1.0 {
        -std::f32::consts::FRAC_PI_2
    } else if value >= 1.0 {
        std::f32::consts::FRAC_PI_2
    } else {
        value.asin()
    }
}

/// `angle_normalized_v3v3`.
fn angle_normalized(first: [f32; 3], second: [f32; 3]) -> f32 {
    let length = |a: [f32; 3], b: [f32; 3]| {
        let d = sub(a, b);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    if dot(first, second) >= 0.0 {
        2.0 * saasin(length(first, second) / 2.0)
    } else {
        let negated = [-second[0], -second[1], -second[2]];
        std::f32::consts::PI - 2.0 * saasin(length(first, negated) / 2.0)
    }
}

/// `angle_v3v3v3`.
fn angle_v3v3v3(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    angle_normalized(normalize(sub(a, b)), normalize(sub(c, b)))
}

/// `cos_v3v3v3`.
fn cos_v3v3v3(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    dot(normalize(sub(a, b)), normalize(sub(c, b)))
}
