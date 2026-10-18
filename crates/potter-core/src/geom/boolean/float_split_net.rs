//! Blender 5.2.2 `BM_face_split_edgenet` for the FLOAT solver.
//!
//! Port of `source/blender/bmesh/intern/bmesh_polygon_edgenet.cc`: `BM_face_split_edgenet` and its
//! `bm_face_split_edgenet_find_loop*` helpers. `face_edges_split` in
//! `source/blender/bmesh/tools/bmesh_intersect.cc` calls it after the island connect. It splits one
//! face into the regions bounded by its boundary and an edge net.
//!
//! # What the port keeps from the `BMesh`
//! - The graph: the face's boundary ring plus the net edges, deduplicated as undirected vertex pairs.
//! - Per vertex, the disk cycle of graph edges and the disk head (`v->e`). The walk overwrites the head
//!   with back-pointers exactly as Blender does, so the head is mutable state for the whole call.
//! - Per edge, the flagged loops (`FACE_NET`): the face being split, tagged [`ORIGINAL`], and each face
//!   emitted so far. The count is Blender's `bm_edge_flagged_radial_count`.
//! - The vertex queue: LIFO, seeded with `boundary[0]`, refilled from each new face's vertices.
//! - `BM_edge_is_wire` and `BM_face_exists`, which need the mesh's other faces (`mesh_faces`).
//!
//! # Inputs
//! - `boundary`: the face's vertex ring in loop order. Loop `i` runs `boundary[i] -> boundary[i + 1]`.
//! - `edge_net`: undirected vertex pairs. Blender's `BMEdge` orientation is never read by this routine,
//!   only the endpoints, so no edge direction is assumed. Duplicates and pairs equal to a boundary edge
//!   are merged, which matches Blender's flag semantics (`EDGE_NET` is idempotent).
//! - Disk cycle order. Blender appends a new edge just before the vertex's head, so the cycle read from
//!   the head is creation order. This port takes creation order to be the boundary edges in ring order,
//!   then the net edges in the given order, and starts each vertex's head at its first edge. Disk order
//!   decides LIFO pops in the pair search and sort ties in the walk. Callers that know the mesh's real
//!   creation order should pass lists in that order.
//!
//! # Output
//! [`SplitNetResult::faces`] lists the emitted faces in Blender's creation order. Each ring starts at the
//! vertex the search began from, which is Blender's `face_verts[0]` and the first loop of the new face.
//! An empty list means no split: Blender keeps the original face unchanged.
//!
//! # Corner provenance
//! [`SplitNetResult::corner_sources`] gives every corner of every emitted face its loop-data source.
//! Blender's rule, all inside `BM_face_split_edgenet` (`bmesh_polygon_edgenet.cc`):
//! - Each new face is `BM_face_create(bm, verts, edges, len, f, BM_CREATE_NOP)` (line 575). Face-domain
//!   attributes come from the split face `f` (`bmesh_core.cc:562-565`). Its loops start at
//!   `CustomData_bmesh_set_default` (`bmesh_core.cc:337-343`), because `bm_loop_create` gets no example
//!   loop.
//! - The loop pass runs only if `CustomData_has_math(&bm->ldata)` (line 611). `bm->ldata` is the union of
//!   every input mesh's corner layers (`geometry_mesh_boolean.cc:987-1000`). Math needs `equal`, `add`,
//!   `multiply`, `initminmax` and `dominmax` (`CustomData_layer_has_math`, `customdata.cc:3777`). Live
//!   types with all five are `CD_PROP_FLOAT2` (`customdata.cc:1913`), `CD_PROP_COLOR` (`:1885`) and
//!   `CD_PROP_BYTE_COLOR` (`:1638`); deprecated `CD_ORIGSPACE_MLOOP` also has them. Without such a layer
//!   no corner receives any data, not even at boundary vertices.
//! - Step 1: a corner at a vertex `v` of `f`'s ring copies `f`'s loop at `v` for every layer
//!   (`bmesh_polygon_edgenet.cc:628-635`; `CustomData_bmesh_copy_block`, `customdata.cc:3716`). This is
//!   [`CornerSource::Original`] with `i` such that `boundary[i] == v`.
//! - Step 2: every other vertex `v` of the edge net gets `CustomData_bmesh_interp` over `f`'s loops with
//!   `interp_weights_poly_v2` weights of `v` in `f`'s 2D frame (`bmesh_polygon_edgenet.cc:652-660`). The
//!   first loop at `v` is interpolated and the others copy it (`:665`). This is
//!   [`CornerSource::Interpolated`], shared by all corners at `v`.
//! - `CustomData_bmesh_interp` writes only layers whose type has an `interp` callback (`customdata.cc:3984`).
//!   `INT8` (`:1869`), `INT16_2D` (`:1692`), `INT32_2D` (`:1877`) and `STRING` (`:1593`) have none, so an
//!   interpolated corner keeps the default for them. The step-2 copy carries that default too.
//! - `BM_face_copy_shared` (`all/bmesh_construct.cc:79`) and `BM_face_interp_from_face`
//!   (`all/bmesh_interp.cc:176`) are not called here. The loop pass inlines the interpolation and only
//!   touches net vertices.
//! - The first emitted face is swapped into `f` (`bmesh_polygon_edgenet.cc:689`, `bmesh_core.cc:2999`).
//!   Its corners are unchanged.
//!
//! Consumers: corner `i` of `faces[k]` takes its data from `corner_sources[k][i]`. An `Original` corner
//! copies the source face's corner at that vertex. An `Interpolated` corner is interpolated from the
//! source face's corners with the weights at the vertex. Ring vertices inserted onto the boundary before
//! the call are in `boundary` and copy. Blender's copy there reads the split-time loop, which this port
//! does not model.
//!
//! # Arithmetic
//! f32, in Blender's operand order: the 2D frame of `axis_dominant_v3_to_m3(f->no)`, `normalize_v2`
//! for search directions, and `angle_signed_on_axis_v3v3v3_v3` for fork order. Blender's arm64 FMA
//! contraction is not reproduced, matching the connect port.
//!
//! # Departures
//! - Blender sorts forks with `qsort`, which is not stable. This port uses a stable sort, so only exact
//!   angle ties can differ.
//! - Blender's `BM_face_exists` scans the mesh. This port scans `boundary`, `mesh_faces`, and the faces
//!   emitted so far, which is the same set when `mesh_faces` holds the rest of the mesh.
//! - Ring reconstruction follows back-pointers until it returns to the start vertex. Blender loops
//!   forever if it never does. This port gives up on that loop after visiting every vertex once.

use std::cmp::Ordering;
use std::collections::HashMap;

type P2 = [f32; 2];
type P3 = [f32; 3];

/// Owner of the flagged loops of the face being split. Blender's pointer test
/// `face_normal == l_walk->f->no` holds only for that face.
const ORIGINAL: usize = usize::MAX;

/// Where one corner of an emitted face takes its loop data from. Blender rule: module doc, "Corner
/// provenance".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CornerSource {
    /// The face's own loop `i`, at `boundary[i]`, which is the same vertex. Copied for every layer.
    Original(usize),
    /// Mean-value interpolation of the face's loops at the vertex, shared by every corner there. Only
    /// layers with an `interp` callback are filled; the others keep their defaults.
    Interpolated,
}

/// Result of splitting one face by an edge net.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct SplitNetResult {
    /// Emitted faces in Blender creation order. Loop `i` runs `ring[i] -> ring[i + 1]`, and each ring starts
    /// at the vertex its search began from. Empty means no split.
    pub(crate) faces: Vec<Vec<usize>>,
    /// Corner sources, parallel to `faces`: `corner_sources[k][i]` describes corner `i` of `faces[k]`.
    pub(crate) corner_sources: Vec<Vec<CornerSource>>,
    /// Net edges that no emitted face uses. Blender leaves these without loops.
    pub(crate) dangling: Vec<(usize, usize)>,
}

/// Blender `BM_face_split_edgenet`, the only entry point. `mesh_faces` are the other faces of the mesh;
///
/// `mesh_faces` must not repeat the face being split. Its rings matter only for `BM_edge_is_wire` and
/// `BM_face_exists`, so they decide whether an identical face already present is kept rather than
/// duplicated.
pub(crate) fn face_split_edgenet_with_mesh(
    boundary: &[usize],
    edge_net: &[(usize, usize)],
    positions: &[P3],
    face_normal: P3,
    mesh_faces: &[Vec<usize>],
) -> SplitNetResult {
    // Blender returns before touching the face when the net is empty.
    let Some(&first) = boundary.first() else {
        return SplitNetResult::default();
    };
    if edge_net.is_empty() {
        return SplitNetResult::default();
    }

    let mut net = Net::new(boundary, edge_net, positions, face_normal, mesh_faces);
    let start = net.local_of[&first];
    net.in_queue[start] = true;
    let mut queue = vec![start];
    let base = net.rings.len();

    // Blender `BM_face_split_edgenet`: pop the LIFO queue, emit the loop found at each vertex, then
    // queue the new face's vertices that still have a boundary-and-wire pair.
    while let Some(v) = queue.pop() {
        net.in_queue[v] = false;
        let Some(found) = net.find_loop(v) else {
            continue;
        };
        let ring_global: Vec<usize> = found.ring.iter().map(|&l| net.ids[l]).collect();
        if found.check_exists && net.face_exists(&ring_global) {
            continue;
        }
        let owner = net.rings.len() - base;
        for (&edge, &from) in found.edges.iter().zip(&found.ring) {
            net.flagged[edge].push((owner, from));
            net.mesh_loops[edge] += 1;
        }
        net.rings.push(ring_global);
        for &w in &found.ring {
            if !net.in_queue[w] && net.has_loop_pair(w) {
                net.in_queue[w] = true;
                queue.push(w);
            }
        }
    }

    let dangling = net
        .edges
        .iter()
        .filter(|&&e| net.flagged[e].is_empty())
        .map(|&e| net.edge_global[e])
        .collect();
    let faces = net.rings.split_off(base);
    let corner_sources = ring_corner_sources(boundary, &faces);
    SplitNetResult {
        faces,
        corner_sources,
        dangling,
    }
}

/// Corner sources of emitted rings, the rule in the module doc's "Corner provenance". A ring vertex
/// that is in `boundary` copies the first boundary loop at that vertex; any other vertex interpolates.
fn ring_corner_sources(boundary: &[usize], faces: &[Vec<usize>]) -> Vec<Vec<CornerSource>> {
    let mut loop_of: HashMap<usize, usize> = HashMap::with_capacity(boundary.len());
    for (i, &v) in boundary.iter().enumerate() {
        loop_of.entry(v).or_insert(i);
    }
    faces
        .iter()
        .map(|ring| {
            ring.iter()
                .map(|v| match loop_of.get(v) {
                    Some(&i) => CornerSource::Original(i),
                    None => CornerSource::Interpolated,
                })
                .collect()
        })
        .collect()
}

/// Undirected edge table in creation order: boundary loops first, then net pairs.
#[derive(Default)]
struct EdgeTable {
    index: HashMap<(usize, usize), usize>,
    /// First-seen orientation of each edge.
    ends: Vec<(usize, usize)>,
}

impl EdgeTable {
    fn intern(&mut self, a: usize, b: usize) -> usize {
        let key = (a.min(b), a.max(b));
        if let Some(&e) = self.index.get(&key) {
            return e;
        }
        let e = self.ends.len();
        self.index.insert(key, e);
        self.ends.push((a, b));
        e
    }

    fn get(&self, a: usize, b: usize) -> Option<usize> {
        self.index.get(&(a.min(b), a.max(b))).copied()
    }
}

/// A closed loop found at one vertex.
struct Loop {
    /// Local vertex ids. `ring[0]` is the search start.
    ring: Vec<usize>,
    /// `edges[j]` joins `ring[j]` and `ring[j + 1]`.
    edges: Vec<usize>,
    /// Blender `check_face_exists`: every edge of the loop already has a loop in the mesh.
    check_exists: bool,
}

/// Mutable state of one `BM_face_split_edgenet` call. Vertices and edges are local indices.
struct Net {
    /// Global vertex id of each local vertex.
    ids: Vec<usize>,
    local_of: HashMap<usize, usize>,
    co: Vec<P3>,
    /// Global endpoints of each graph edge, in first-seen orientation.
    edge_global: Vec<(usize, usize)>,
    /// Local endpoints of each graph edge.
    ends: Vec<[usize; 2]>,
    /// Net edges, deduplicated, in input order. Boundary edges are excluded.
    edges: Vec<usize>,
    /// Disk cycle per local vertex: its graph edges in cyclic order.
    disk: Vec<Vec<usize>>,
    /// Position of each edge in `disk[ends[e][0]]` and `disk[ends[e][1]]`.
    slot_pos: Vec<[usize; 2]>,
    /// Blender `v->e` as a position in `disk[v]`.
    head: Vec<usize>,
    /// Flagged loops per edge: `(owner, loop start vertex)`. `owner` is `ORIGINAL` or an emitted face index.
    flagged: Vec<Vec<(usize, usize)>>,
    /// Loops in the mesh per edge, including the boundary face and `mesh_faces`. Zero means wire.
    mesh_loops: Vec<usize>,
    /// Global rings of every face the mesh holds: boundary, `mesh_faces`, then emitted faces.
    rings: Vec<Vec<usize>>,
    in_queue: Vec<bool>,
    /// Blender `VERT_VISIT` on fork vertices, set during one walk and cleared at its end.
    visit: Vec<bool>,
    normal: P3,
    /// `axis_dominant_v3_to_m3(normal)`: the two in-plane basis vectors whose dot products give 2D.
    b0: P3,
    b1: P3,
}

impl Net {
    fn new(
        boundary: &[usize],
        edge_net: &[(usize, usize)],
        positions: &[P3],
        normal: P3,
        mesh_faces: &[Vec<usize>],
    ) -> Self {
        let n = boundary.len();
        let mut ids: Vec<usize> = boundary
            .iter()
            .copied()
            .chain(edge_net.iter().flat_map(|&(a, b)| [a, b]))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        let local_of: HashMap<usize, usize> =
            ids.iter().enumerate().map(|(i, &g)| (g, i)).collect();

        let mut table = EdgeTable::default();
        let loop_edge: Vec<usize> = (0..n)
            .map(|i| table.intern(boundary[i], boundary[(i + 1) % n]))
            .collect();
        let net_intern: Vec<usize> = edge_net
            .iter()
            .map(|&(a, b)| {
                debug_assert!(a != b, "a net edge joins two distinct vertices");
                table.intern(a, b)
            })
            .collect();
        let m = table.ends.len();

        let mut is_boundary = vec![false; m];
        for &e in &loop_edge {
            is_boundary[e] = true;
        }
        let mut edges = Vec::new();
        let mut seen = vec![false; m];
        for &e in &net_intern {
            if !is_boundary[e] && !seen[e] {
                seen[e] = true;
                edges.push(e);
            }
        }

        let ends: Vec<[usize; 2]> = table
            .ends
            .iter()
            .map(|&(a, b)| [local_of[&a], local_of[&b]])
            .collect();

        // Disk cycles in creation order; the head starts at each vertex's first edge.
        let mut disk: Vec<Vec<usize>> = vec![Vec::new(); ids.len()];
        let mut slot_pos = vec![[0usize; 2]; m];
        for (e, &[a, b]) in ends.iter().enumerate() {
            slot_pos[e][0] = disk[a].len();
            disk[a].push(e);
            slot_pos[e][1] = disk[b].len();
            disk[b].push(e);
        }
        let head = vec![0; ids.len()];

        let mut flagged: Vec<Vec<(usize, usize)>> = vec![Vec::new(); m];
        let mut mesh_loops = vec![0usize; m];
        for (i, &e) in loop_edge.iter().enumerate() {
            flagged[e].push((ORIGINAL, local_of[&boundary[i]]));
            mesh_loops[e] += 1;
        }
        for ring in mesh_faces {
            let len = ring.len();
            for (i, &a) in ring.iter().enumerate() {
                if let Some(e) = table.get(a, ring[(i + 1) % len]) {
                    mesh_loops[e] += 1;
                }
            }
        }

        let mut rings = Vec::with_capacity(1 + mesh_faces.len());
        rings.push(boundary.to_vec());
        rings.extend(mesh_faces.iter().cloned());

        let co: Vec<P3> = ids.iter().map(|&g| positions[g]).collect();
        let (b0, b1) = ortho_basis(normal);
        let nverts = ids.len();
        Self {
            ids,
            local_of,
            co,
            edge_global: table.ends,
            ends,
            edges,
            disk,
            slot_pos,
            head,
            flagged,
            mesh_loops,
            rings,
            in_queue: vec![false; nverts],
            visit: vec![false; nverts],
            normal,
            b0,
            b1,
        }
    }

    /// Blender `BM_edge_other_vert`.
    fn other(&self, e: usize, v: usize) -> usize {
        let [a, b] = self.ends[e];
        if a == v { b } else { a }
    }

    /// Position of edge `e` in `disk[v]`, for `v` an endpoint of `e`.
    fn pos_in(&self, e: usize, v: usize) -> usize {
        if self.ends[e][0] == v {
            self.slot_pos[e][0]
        } else {
            self.slot_pos[e][1]
        }
    }

    /// Blender `v->e`.
    fn head_edge(&self, v: usize) -> usize {
        self.disk[v][self.head[v]]
    }

    /// Blender `v->e = e`, stored as a disk position.
    fn set_head(&mut self, v: usize, e: usize) {
        self.head[v] = self.pos_in(e, v);
    }

    /// Edges at `v` in disk order from its head, split into boundary edges (one flagged loop) and
    /// wire edges (none). Each list keeps iteration order, so its last element is the LIFO top.
    fn classify(&self, v: usize) -> (Vec<usize>, Vec<usize>) {
        let disk = &self.disk[v];
        let len = disk.len();
        let mut boundary = Vec::new();
        let mut wire = Vec::new();
        for k in 0..len {
            let e = disk[(self.head[v] + k) % len];
            match self.flagged[e].len() {
                1 => boundary.push(e),
                0 => wire.push(e),
                _ => {}
            }
        }
        (boundary, wire)
    }

    /// Blender `bm_face_split_edgenet_find_loop_pair_exists`.
    fn has_loop_pair(&self, v: usize) -> bool {
        let (boundary, wire) = self.classify(v);
        !boundary.is_empty() && (!wire.is_empty() || boundary.len() >= 2)
    }

    /// Blender `normalize_v2_m3_v3v3(out, face_normal_matrix, co[to], co[from])`.
    fn direction(&self, to: usize, from: usize) -> P2 {
        normalize2(proj2(self.b0, self.b1, sub3(self.co[to], self.co[from])))
    }

    /// Blender `bm_face_split_edgenet_find_loop_pair`. Returns `[first, second]` with the first edge
    /// oriented for the walk.
    fn loop_pair(&self, v: usize) -> Option<[usize; 2]> {
        let (mut boundary, mut wire) = self.classify(v);
        let boundary_len = boundary.len();
        let mut e_pair = [boundary.pop()?, 0];
        let mut search = if wire.is_empty() {
            if boundary_len < 2 {
                return None;
            }
            e_pair[1] = boundary.pop()?;
            boundary
        } else {
            e_pair[1] = wire.pop()?;
            wire
        };

        // Among the remaining candidates, take the one that bends least from `e_pair[0]`.
        if !search.is_empty() {
            let v_prev = self.other(e_pair[0], v);
            let v_next = self.other(e_pair[1], v);
            let dir_prev = self.direction(v_prev, v);
            let dir_next = self.direction(v_next, v);
            let mut best = dot2(dir_next, dir_prev);
            while let Some(e) = search.pop() {
                let dir_test = self.direction(self.other(e, v), v);
                let cos = dot2(dir_prev, dir_test);
                if cos > best {
                    best = cos;
                    e_pair[1] = e;
                }
            }
        }

        // Wind by the boundary edge's flagged loop: its owner and the loop's start vertex.
        let &(owner, start) = self.flagged[e_pair[0]].first()?;
        let mut swap = false;
        if owner == ORIGINAL {
            swap = !swap;
        }
        if start != v {
            swap = !swap;
        }
        if swap {
            e_pair.swap(0, 1);
        }
        Some(e_pair)
    }

    /// Blender `bm_face_split_edgenet_find_loop_walk`'s child collection: the vertices reachable from `v`
    /// over graph edges with at most one flagged loop, skipping the back-pointer edge. Each collected
    /// vertex gets its back-pointer set to the edge it was reached by.
    fn children(&mut self, v: usize) -> Vec<usize> {
        let len = self.disk[v].len();
        let order: Vec<usize> = (1..len)
            .map(|k| self.disk[v][(self.head[v] + k) % len])
            .collect();
        let mut out = Vec::new();
        for e in order {
            if self.flagged[e].len() < 2 {
                let w = self.other(e, v);
                if !self.visit[w] {
                    self.set_head(w, e);
                    out.push(w);
                }
            }
        }
        out
    }

    /// Blender `bm_face_split_edgenet_find_loop_walk`: a depth-first walk over the net from the first
    /// boundary neighbour of `v_init`, ordering forks by signed angle. Returns whether the second edge's
    /// far vertex was reached. Back-pointers left in `head` describe the path.
    fn walk(&mut self, v_init: usize, e_pair: [usize; 2]) -> bool {
        let v_dst = self.other(e_pair[1], v_init);
        let v0 = self.other(e_pair[0], v_init);
        self.set_head(v0, e_pair[0]);

        // Blender keeps two LIFO stacks and swaps them after each expansion. Preserve that exactly.
        let mut stack = vec![v0];
        let mut next: Vec<usize> = Vec::new();
        let mut tagged: Vec<usize> = Vec::new();
        let mut found = false;

        'pop: while let Some(mut v) = stack.pop() {
            loop {
                if v == v_dst {
                    found = true;
                    break 'pop;
                }
                let children = self.children(v);
                if let &[only] = children.as_slice() {
                    // Single child: step without tagging, as Blender's no-fork fast path does.
                    v = only;
                    continue;
                }
                if children.len() > 1 {
                    let v_prev = self.other(self.head_edge(v), v);
                    let mut order: Vec<(f32, usize)> = children
                        .iter()
                        .map(|&w| {
                            let angle = angle_signed_on_axis(
                                self.co[v_prev],
                                self.co[v],
                                self.co[w],
                                self.normal,
                            );
                            (angle, w)
                        })
                        .collect();
                    // `BLI_sortutil_cmp_float_reverse`: largest angle first.
                    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
                    self.visit[v] = true;
                    tagged.push(v);
                    next.extend(order.iter().rev().map(|&(_, w)| w));
                }
                break;
            }
            if !next.is_empty() {
                std::mem::swap(&mut stack, &mut next);
            }
        }

        for v in tagged {
            self.visit[v] = false;
        }
        found
    }

    /// Blender `bm_face_split_edgenet_find_loop`: pick a pair at `v_init`, walk to close the loop, and
    /// rebuild the ring from back-pointers.
    fn find_loop(&mut self, v_init: usize) -> Option<Loop> {
        let e_pair = self.loop_pair(v_init)?;
        if !self.walk(v_init, e_pair) {
            return None;
        }

        let mut ring = vec![v_init];
        let mut edges = vec![e_pair[1]];
        let mut v = self.other(e_pair[1], v_init);
        loop {
            ring.push(v);
            edges.push(self.head_edge(v));
            v = self.other(self.head_edge(v), v);
            if v == v_init {
                break;
            }
            if ring.len() > self.co.len() {
                return None;
            }
        }
        if ring.len() < 3 {
            return None;
        }
        // `BM_edge_is_wire` is false for each edge, so `BM_face_exists` applies.
        let check_exists = edges.iter().all(|&e| self.mesh_loops[e] > 0);
        Some(Loop {
            ring,
            edges,
            check_exists,
        })
    }

    /// Blender `BM_face_exists`: a face of the mesh with this vertex cycle, in either direction.
    fn face_exists(&self, cand: &[usize]) -> bool {
        self.rings.iter().any(|ring| ring_matches(ring, cand))
    }
}

/// Blender `BM_face_exists` on one face: a loop at `cand[0]` followed by `cand[1]` (forwards or
/// backwards, as Blender tries them) with the same vertices after it.
fn ring_matches(ring: &[usize], cand: &[usize]) -> bool {
    let n = ring.len();
    if n != cand.len() || n < 2 {
        return false;
    }
    for (l, &v) in ring.iter().enumerate() {
        if v != cand[0] {
            continue;
        }
        let step: isize = if ring[(l + 1) % n] == cand[1] {
            1
        } else if ring[(l + n - 1) % n] == cand[1] {
            -1
        } else {
            continue;
        };
        let matched = (2..n).all(|i| {
            let idx = (l as isize + step * i as isize).rem_euclid(n as isize) as usize;
            ring[idx] == cand[i]
        });
        if matched {
            return true;
        }
    }
    false
}

/// Blender `ortho_basis_v3v3_v3`: the two tangents of `n`, in the order used by `axis_dominant_v3_to_m3`.
fn ortho_basis(n: P3) -> (P3, P3) {
    let f = n[0] * n[0] + n[1] * n[1];
    if f > f32::EPSILON {
        let d = 1.0 / f.sqrt();
        let n1 = [n[1] * d, -n[0] * d, 0.0];
        let n2 = [-n[2] * n1[1], n[2] * n1[0], n[0] * n1[1] - n[1] * n1[0]];
        (n1, n2)
    } else {
        let n1 = [if n[2] < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0];
        (n1, [0.0, 1.0, 0.0])
    }
}

/// Blender `mul_v2_m3v3` with `M = axis_dominant_v3_to_m3(n)`: the coordinates of `a` along the two
/// tangents.
fn proj2(b0: P3, b1: P3, a: P3) -> P2 {
    [dot3(b0, a), dot3(b1, a)]
}

/// Blender `normalize_v2`.
fn normalize2(a: P2) -> P2 {
    let d = a[0] * a[0] + a[1] * a[1];
    if d > 1.0e-35 {
        let d = d.sqrt();
        let m = 1.0 / d;
        [a[0] * m, a[1] * m]
    } else {
        [0.0, 0.0]
    }
}

/// Blender `normalize_v3_v3`.
fn normalize3(a: P3) -> P3 {
    let d = dot3(a, a);
    if d > 1.0e-35 {
        let d = d.sqrt();
        let m = 1.0 / d;
        [a[0] * m, a[1] * m, a[2] * m]
    } else {
        [0.0; 3]
    }
}

fn dot2(a: P2, b: P2) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn dot3(a: P3, b: P3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub3(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Blender `len_v3v3`.
fn len_v3v3(a: P3, b: P3) -> f32 {
    let d = sub3(b, a);
    dot3(d, d).sqrt()
}

/// Blender `safe_asinf`.
fn safe_asin(x: f32) -> f32 {
    x.clamp(-1.0, 1.0).asin()
}

/// Blender `angle_normalized_v3v3`.
fn angle_normalized(v1: P3, v2: P3) -> f32 {
    if dot3(v1, v2) >= 0.0 {
        2.0 * safe_asin(len_v3v3(v1, v2) / 2.0)
    } else {
        let v2n = [-v2[0], -v2[1], -v2[2]];
        std::f32::consts::PI - 2.0 * safe_asin(len_v3v3(v1, v2n) / 2.0)
    }
}

/// Blender `angle_v3v3`.
fn angle_v3v3(a: P3, b: P3) -> f32 {
    angle_normalized(normalize3(a), normalize3(b))
}

/// Blender `project_plane_normalized_v3_v3v3`.
fn project_plane_normalized(p: P3, plane: P3) -> P3 {
    let mul = dot3(p, plane);
    let neg = -mul;
    [
        p[0] + plane[0] * neg,
        p[1] + plane[1] * neg,
        p[2] + plane[2] * neg,
    ]
}

/// Blender `angle_signed_on_axis_v3v3_v3`.
fn angle_signed_on_axis_vecs(v1: P3, v2: P3, axis: P3) -> f32 {
    let v1_proj = project_plane_normalized(v1, axis);
    let v2_proj = project_plane_normalized(v2, axis);
    let mut angle = angle_v3v3(v1_proj, v2_proj);
    let t = cross3(v2_proj, v1_proj);
    if dot3(t, axis) < 0.0 {
        angle = std::f32::consts::TAU - angle;
    }
    angle
}

/// Blender `angle_signed_on_axis_v3v3v3_v3`: the signed angle at `v2` from `v1` to `v3`.
fn angle_signed_on_axis(v1: P3, v2: P3, v3: P3, axis: P3) -> f32 {
    angle_signed_on_axis_vecs(sub3(v1, v2), sub3(v3, v2), axis)
}

#[cfg(test)]
mod tests {
    use super::CornerSource::{Interpolated, Original};
    use super::*;

    /// Square of side 2 with an interior point at its centre (index 4).
    const POINTS: [P3; 5] = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    const BOUNDARY: [usize; 4] = [0, 1, 2, 3];
    const UP: P3 = [0.0, 0.0, 1.0];

    #[test]
    fn split_net_diagonal_makes_two_triangles() {
        let result = face_split_edgenet_with_mesh(&BOUNDARY, &[(0, 2)], &POINTS, UP, &[]);
        assert_eq!(result.faces, vec![vec![0, 2, 3], vec![2, 0, 1]]);
        assert_eq!(result.dangling, Vec::<(usize, usize)>::new());
    }

    #[test]
    fn split_net_interior_spur_keeps_face_and_dangles() {
        // The spur leaves the search at corner 0 with only a two-vertex loop, so Blender emits nothing
        // and the original square stays. The spur has no loop and is reported as dangling.
        let result = face_split_edgenet_with_mesh(&BOUNDARY, &[(0, 4)], &POINTS, UP, &[]);
        assert_eq!(result.faces, Vec::<Vec<usize>>::new());
        assert_eq!(result.dangling, vec![(0, 4)]);
    }

    #[test]
    fn split_net_cross_makes_four_triangles() {
        let net = [(0, 4), (4, 2), (1, 4), (4, 3)];
        let result = face_split_edgenet_with_mesh(&BOUNDARY, &net, &POINTS, UP, &[]);
        assert_eq!(
            result.faces,
            vec![vec![0, 4, 3], vec![3, 4, 2], vec![2, 4, 1], vec![1, 4, 0]]
        );
        assert_eq!(result.dangling, Vec::<(usize, usize)>::new());
    }

    #[test]
    fn split_net_existing_identical_face_is_not_duplicated() {
        // The mesh already holds triangle (0, 2, 3), so Blender's `BM_face_exists` skips the loop.
        let existing = vec![vec![0, 2, 3]];
        let result = face_split_edgenet_with_mesh(&BOUNDARY, &[(0, 2)], &POINTS, UP, &existing);
        assert_eq!(result.faces, Vec::<Vec<usize>>::new());
        assert_eq!(result.dangling, vec![(0, 2)]);
    }

    #[test]
    fn split_net_empty_net_keeps_face() {
        let result = face_split_edgenet_with_mesh(&BOUNDARY, &[], &POINTS, UP, &[]);
        assert_eq!(result.faces, Vec::<Vec<usize>>::new());
        assert_eq!(result.dangling, Vec::<(usize, usize)>::new());
    }

    #[test]
    fn split_net_corner_diagonal_copies_original_loops() {
        // Every corner is a square vertex, so each copies the original face's loop at that vertex
        // (Blender `bmesh_polygon_edgenet.cc:628-635`). The faces are the diagonal test's.
        let result = face_split_edgenet_with_mesh(&BOUNDARY, &[(0, 2)], &POINTS, UP, &[]);
        assert_eq!(
            result.corner_sources,
            vec![
                vec![Original(0), Original(2), Original(3)],
                vec![Original(2), Original(0), Original(1)],
            ]
        );
    }

    #[test]
    fn split_net_corner_centre_is_interpolated() {
        // The centre (4) is not a boundary vertex, so its corners interpolate (`:652-665`). Each
        // triangle copies its two square corners and interpolates the centre corner.
        let net = [(0, 4), (4, 2), (1, 4), (4, 3)];
        let result = face_split_edgenet_with_mesh(&BOUNDARY, &net, &POINTS, UP, &[]);
        assert_eq!(result.corner_sources.len(), result.faces.len());
        for (ring, sources) in result.faces.iter().zip(&result.corner_sources) {
            assert_eq!(ring.len(), sources.len());
        }
        assert_eq!(
            result.corner_sources,
            vec![
                vec![Original(0), Interpolated, Original(3)],
                vec![Original(3), Interpolated, Original(2)],
                vec![Original(2), Interpolated, Original(1)],
                vec![Original(1), Interpolated, Original(0)],
            ]
        );
    }

    #[test]
    fn split_net_corner_index_is_boundary_position_not_vertex_id() {
        // The diagonal square with boundary ids 5, 7, 2, 4. Faces are the diagonal test's faces with
        // local ids 0, 1, 2, 3 mapped to 5, 7, 2, 4. Loop `i` is `boundary[i]`, so vertex 2 is loop 2.
        let mut positions = vec![[0.0_f32; 3]; 8];
        positions[5] = [0.0, 0.0, 0.0];
        positions[7] = [2.0, 0.0, 0.0];
        positions[2] = [2.0, 2.0, 0.0];
        positions[4] = [0.0, 2.0, 0.0];
        let result = face_split_edgenet_with_mesh(&[5, 7, 2, 4], &[(5, 2)], &positions, UP, &[]);
        assert_eq!(result.faces, vec![vec![5, 2, 4], vec![2, 5, 7]]);
        assert_eq!(
            result.corner_sources,
            vec![
                vec![Original(0), Original(2), Original(3)],
                vec![Original(2), Original(0), Original(1)],
            ]
        );
    }

    #[test]
    fn split_net_corner_no_split_has_no_sources() {
        // Blender keeps the original face when nothing is emitted, so there are no corners to describe.
        let spur = face_split_edgenet_with_mesh(&BOUNDARY, &[(0, 4)], &POINTS, UP, &[]);
        let existing = vec![vec![0, 2, 3]];
        let duplicate = face_split_edgenet_with_mesh(&BOUNDARY, &[(0, 2)], &POINTS, UP, &existing);
        let empty = face_split_edgenet_with_mesh(&BOUNDARY, &[], &POINTS, UP, &[]);
        assert_eq!(spur.corner_sources.len(), 0);
        assert_eq!(duplicate.corner_sources.len(), 0);
        assert_eq!(empty.corner_sources.len(), 0);
    }
}
