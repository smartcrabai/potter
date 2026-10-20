//! UV sphere storage order of Blender 5.2.2 `primitive_uv_sphere_add`.
//!
//! Blender builds the sphere in `bmo_create_uvsphere_exec` (`bmo_primitive.cc:845-963`): one seed
//! meridian of `rings + 1` vertices and `rings` edges, then `segments` rounds of
//! `extrude_edge_only` followed by a rotation of the new vertices about the sphere axis, then
//! `remove_doubles` at a third of the shortest segment. This module replays that pipeline on a
//! minimal `BMesh` model, so the vertex, edge, face and loop order follow from the same rules:
//!
//! - Positions are computed in f32 with Blender's operation order (`sin_cos_from_fraction`,
//!   `axis_angle_normalized_to_mat3_ex`, `mul_m4_m4m4` C fallback, `mul_m4_v3`). The f32 rounding
//!   decides which copy of a coincident pair survives, so the arithmetic is not optional.
//! - `remove_doubles` uses `bmesh_find_doubles_by_distance_impl` (`bmo_removedoubles.cc`) over a
//!   balanced kd-tree (`BLI_kdtree.hh`). Every cluster includes its search origin twice, so the
//!   centroid rule picks the origin among pairs, and the surviving copy is the one first in tree
//!   node order.
//! - Duplicated vertices are copied in source order and each boundary edge gets one quad
//!   (`bmo_extrude.cc`). Vertices and edges created while welding are appended in mempool order.
//!   Face slots are reused in place, so the face list keeps creation order.
//! - Edges of a new face are created from the last vertex to the first (`BM_edges_from_verts_ensure`,
//!   `bmesh_construct.cc:57`), which fixes the stored edge direction.
//!
//! Blender's extrusion loops over `boundary_map.out`, a `GHash` keyed by `BMEdge *`
//! (`bmesh_operators.cc:100`, `bmo_extrude.cc`). Its iteration order follows pointer hashes, so
//! the face order inside each band differs between Blender processes; the vertex order does not.
//! Potter uses the boundary edges' creation order, which is the order Blender produces when the
//! boundary map iterates in edge-creation order. Vertex order matches Blender bit for bit;
//! edge and face order match up to that per-band permutation.

use std::collections::HashMap;

use glam::DVec3;

use super::{Edge, Face, Mesh, MeshError, UvSphereParams, allocate};

/// `NONE` marks a missing link or an unassigned duplicate slot.
const NONE: usize = usize::MAX;

type Vec3 = [f32; 3];
type Mat3 = [[f32; 3]; 3];
type Mat4 = [[f32; 4]; 4];

/// `sin_cos_from_fraction` (`math_rotation_c.cc:995`): sine and cosine of `2π·numer/denom`,
/// evaluated in the first octant and mapped back by symmetry for exact symmetric circles.
/// Compacted vertex and edge indices of a welded cluster.
type Compacted = (Vec<usize>, Vec<usize>);

/// Bit-exact equality of two non-negative squared distances: the tie-break compares computed values, so
/// equal bits are the intended test.
fn same_distance(first: f32, second: f32) -> bool {
    first.to_bits() == second.to_bits()
}

fn sin_cos_from_fraction(numerator: i32, denominator: i32) -> (f32, f32) {
    let mut numerator = numerator * 8;
    let octant = numerator / denominator;
    let denominator = denominator * 8;
    let mut cos_sign = 1.0_f32;
    let mut swap = false;
    match octant {
        0 => {}
        1 | 2 => {
            numerator = denominator / 4 - numerator;
            swap = true;
        }
        3 | 4 => {
            numerator = denominator / 2 - numerator;
            cos_sign = -1.0;
        }
        5 | 6 => {
            numerator -= denominator * 3 / 4;
            swap = true;
            cos_sign = -1.0;
        }
        _ => numerator -= denominator,
    }
    let angle = (2.0 * std::f64::consts::PI) as f32 * (numerator as f32 / denominator as f32);
    let sin = angle.sin();
    let cos = angle.cos() * cos_sign;
    if swap { (cos, sin) } else { (sin, cos) }
}

/// `axis_angle_normalized_to_mat3_ex` (`math_rotation_c.cc:1138`) for a unit axis.
fn axis_angle_mat3(axis: Vec3, angle: f32) -> Mat3 {
    let sin = angle.sin();
    let cos = angle.cos();
    let ico = 1.0 - cos;
    let nsi = [axis[0] * sin, axis[1] * sin, axis[2] * sin];
    let n00 = (axis[0] * axis[0]) * ico;
    let n01 = (axis[0] * axis[1]) * ico;
    let n11 = (axis[1] * axis[1]) * ico;
    let n02 = (axis[0] * axis[2]) * ico;
    let n12 = (axis[1] * axis[2]) * ico;
    let n22 = (axis[2] * axis[2]) * ico;
    [
        [n00 + cos, n01 + nsi[2], n02 - nsi[1]],
        [n01 - nsi[2], n11 + cos, n12 + nsi[0]],
        [n02 + nsi[1], n12 - nsi[0], n22 + cos],
    ]
}

/// `mul_m4_m4m4` C fallback: `R[i][j] = B[i][0]·A[0][j] + … + B[i][3]·A[3][j]`.
fn mul_m4_m4m4(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut result = [[0.0_f32; 4]; 4];
    for (i, row) in result.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = b[i][0] * a[0][j] + b[i][1] * a[1][j] + b[i][2] * a[2][j] + b[i][3] * a[3][j];
        }
    }
    result
}

/// `mul_m4_v3`: transforms a point by a 4x4 affine matrix.
fn mul_m4_v3(m: &Mat4, v: Vec3) -> Vec3 {
    let [x, y, z] = v;
    [
        x * m[0][0] + y * m[1][0] + m[2][0] * z + m[3][0],
        x * m[0][1] + y * m[1][1] + m[2][1] * z + m[3][1],
        x * m[0][2] + y * m[1][2] + m[2][2] * z + m[3][2],
    ]
}

/// The rotation the extrude step applies: `transform_pivot_set_m4` around `pivot`.
fn rotation_about_pivot(m3: &Mat3, pivot: Vec3) -> Mat4 {
    let mut mat = [[0.0_f32; 4]; 4];
    for (row, source) in mat.iter_mut().zip(m3) {
        row[..3].copy_from_slice(source);
    }
    mat[3][3] = 1.0;
    let mut tmat = [[0.0_f32; 4]; 4];
    for (i, row) in tmat.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    tmat[3][..3].copy_from_slice(&pivot);
    let mat = mul_m4_m4m4(&tmat, &mat);
    tmat[3][..3].copy_from_slice(&[-pivot[0], -pivot[1], -pivot[2]]);
    mul_m4_m4m4(&mat, &tmat)
}

/// `mul_v3_m3v3`, used only for the merge distance.
fn mul_v3_m3v3(m: &Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}
/// Squared distance between two points, summed as `dx² + dy² + dz²`.
fn squared_distance(a: Vec3, b: Vec3) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// Edge key with endpoints in ascending order.
fn pair_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

#[derive(Clone, Copy)]
struct KdNode {
    index: usize,
    co: Vec3,
    d: usize,
    left: usize,
    right: usize,
}

/// `kdtree_balance`: median partition on cycling axes, in place.
fn kd_balance(nodes: &mut [KdNode], axis: usize, offset: usize) -> usize {
    let len = nodes.len();
    match len {
        0 => return NONE,
        1 => return offset,
        _ => {}
    }
    let median = len / 2;
    let mut left = 0;
    let mut right = len - 1;
    while right > left {
        let pivot = nodes[right].co[axis];
        let mut i = left.wrapping_sub(1);
        let mut j = right;
        loop {
            i = i.wrapping_add(1);
            while nodes[i].co[axis] < pivot {
                i += 1;
            }
            j -= 1;
            while nodes[j].co[axis] > pivot && j > left {
                j -= 1;
            }
            if i >= j {
                break;
            }
            nodes.swap(i, j);
        }
        nodes.swap(i, right);
        if i >= median {
            right = i - 1;
        }
        if i <= median {
            left = i + 1;
        }
    }
    let next_axis = (axis + 1) % 3;
    nodes[median].d = axis;
    let (lower, rest) = nodes.split_at_mut(median);
    let (middle, upper) = rest.split_at_mut(1);
    middle[0].left = kd_balance(lower, next_axis, offset);
    middle[0].right = kd_balance(upper, next_axis, offset + median + 1);
    offset + median
}

/// `kdtree_range_search_cb`: every node within `range` of `co`, search origin included.
fn kd_range(nodes: &[KdNode], root: usize, co: Vec3, range: f32, found: &mut Vec<usize>) {
    if root == NONE {
        return;
    }
    let range_sq = range * range;
    let mut stack = vec![root];
    while let Some(position) = stack.pop() {
        let node = &nodes[position];
        let axis = node.d;
        if co[axis] + range < node.co[axis] {
            if node.left != NONE {
                stack.push(node.left);
            }
        } else if co[axis] - range > node.co[axis] {
            if node.right != NONE {
                stack.push(node.right);
            }
        } else {
            if squared_distance(node.co, co) <= range_sq {
                found.push(node.index);
            }
            if node.left != NONE {
                stack.push(node.left);
            }
            if node.right != NONE {
                stack.push(node.right);
            }
        }
    }
}

/// `deduplicate_target_calc_fn`: the surviving index of a cluster.
fn cluster_target(cluster: &[usize], coords: &[Vec3]) -> usize {
    if cluster.len() == 2 {
        return usize::from(cluster[0] >= cluster[1]);
    }
    let count = cluster.len() as f32;
    let mut centroid = [0.0_f32; 3];
    for &vertex in cluster {
        for (sum, value) in centroid.iter_mut().zip(coords[vertex]) {
            *sum += value;
        }
    }
    for value in &mut centroid {
        *value /= count;
    }
    let end = cluster.len() - 1;
    let mut best_index = end;
    let mut best = squared_distance(centroid, coords[cluster[end]]);
    for (index, &vertex) in cluster[..end].iter().enumerate() {
        let distance = squared_distance(centroid, coords[vertex]);
        if distance > best {
            continue;
        }
        if same_distance(distance, best) && vertex > cluster[best_index] {
            continue;
        }
        best_index = index;
        best = distance;
    }
    best_index
}

/// `bmesh_find_doubles_by_distance_impl`: for every vertex, the index it merges into (itself when kept).
fn find_duplicates(coords: &[Vec3], range: f32) -> Vec<usize> {
    let mut nodes: Vec<KdNode> = coords
        .iter()
        .enumerate()
        .map(|(index, &co)| KdNode {
            index,
            co,
            d: 0,
            left: NONE,
            right: NONE,
        })
        .collect();
    let root = kd_balance(&mut nodes, 0, 0);
    let order: Vec<(usize, Vec3)> = nodes.iter().map(|node| (node.index, node.co)).collect();
    let mut duplicates = vec![NONE; coords.len()];
    let mut found = Vec::new();
    for (origin, co) in order {
        if duplicates[origin] != NONE {
            continue;
        }
        found.clear();
        kd_range(&nodes, root, co, range, &mut found);
        let mut cluster: Vec<usize> = found
            .iter()
            .copied()
            .filter(|&vertex| duplicates[vertex] == NONE)
            .collect();
        if cluster.is_empty() {
            continue;
        }
        cluster.push(origin);
        let target = cluster[cluster_target(&cluster, coords)];
        for vertex in cluster {
            duplicates[vertex] = target;
        }
    }
    duplicates
}

#[derive(Clone)]
struct BEdge {
    v: [usize; 2],
    /// First radial loop as (face, loop index); the loop's vertex is `face.v[index]`.
    l: Option<(usize, usize)>,
    dead: bool,
    collapsed: bool,
}

#[derive(Clone)]
struct BFace {
    v: Vec<usize>,
    e: Vec<usize>,
    dead: bool,
}

/// Minimal `BMesh`: vertices, edges and faces in creation (mempool) order.
#[derive(Default)]
struct BMesh {
    co: Vec<Vec3>,
    edges: Vec<BEdge>,
    faces: Vec<BFace>,
    by_pair: HashMap<(usize, usize), usize>,
}

impl BMesh {
    fn vertex(&mut self, co: Vec3) -> usize {
        self.co.push(co);
        self.co.len() - 1
    }

    fn edge_find(&self, a: usize, b: usize) -> Option<usize> {
        self.by_pair.get(&pair_key(a, b)).copied()
    }

    /// `BM_edge_create` without the duplicate check.
    fn edge_add(&mut self, a: usize, b: usize) -> usize {
        let id = self.edges.len();
        self.edges.push(BEdge {
            v: [a, b],
            l: None,
            dead: false,
            collapsed: false,
        });
        self.by_pair.insert(pair_key(a, b), id);
        id
    }

    /// `BM_edge_create` with `BM_CREATE_NO_DOUBLE`.
    fn edge_ensure(&mut self, a: usize, b: usize) -> usize {
        match self.edge_find(a, b) {
            Some(edge) => edge,
            None => self.edge_add(a, b),
        }
    }

    /// `BM_face_create_verts`: edges are ensured from the last vertex to the first.
    fn face_from_verts(&mut self, verts: &[usize]) -> usize {
        let n = verts.len();
        let mut edges = vec![NONE; n];
        let mut previous = n - 1;
        for current in 0..n {
            edges[previous] = self.edge_ensure(verts[previous], verts[current]);
            previous = current;
        }
        self.face_add(verts.to_vec(), edges)
    }

    fn face_add(&mut self, verts: Vec<usize>, edges: Vec<usize>) -> usize {
        let id = self.faces.len();
        for (index, &edge) in edges.iter().enumerate() {
            if self.edges[edge].l.is_none() {
                self.edges[edge].l = Some((id, index));
            }
        }
        self.faces.push(BFace {
            v: verts,
            e: edges,
            dead: false,
        });
        id
    }
}

/// One `extrude_edge_only` round (`bmo_extrude.cc`) over `input`, followed by its rotation.
///
/// Vertices are duplicated in source order, then edges in source order, then one quad per
/// boundary edge in creation order (the canonical boundary order). Returns the duplicated edges,
/// which form the next round's input.
fn extrude_edge_only(bm: &mut BMesh, input: &[usize], rotation: &Mat4) -> Vec<usize> {
    let mut sources: Vec<usize> = input.iter().flat_map(|&edge| bm.edges[edge].v).collect();
    sources.sort_unstable();
    sources.dedup();
    let mut copy_of: HashMap<usize, usize> = HashMap::with_capacity(sources.len());
    for source in sources {
        let co = mul_m4_v3(rotation, bm.co[source]);
        copy_of.insert(source, bm.vertex(co));
    }
    let duplicates: Vec<usize> = input
        .iter()
        .map(|&edge| {
            let [a, b] = bm.edges[edge].v;
            bm.edge_add(copy_of[&a], copy_of[&b])
        })
        .collect();
    for (&edge, &copy) in input.iter().zip(&duplicates) {
        let [v1, v2] = bm.edges[edge].v;
        let [n1, n2] = bm.edges[copy].v;
        // edge_normal_flip: a boundary edge with a face keeps that face's winding; a seed edge has none.
        let flip = match bm.edges[edge].l {
            None => true,
            Some((face, index)) => v1 == bm.faces[face].v[index],
        };
        let quad = if flip {
            [v2, v1, n1, n2]
        } else {
            [v1, v2, n2, n1]
        };
        bm.face_from_verts(&quad);
    }
    duplicates
}

/// `remdoubles_createface`: the face rebuilt with merged corners, or `None` when it degenerates.
fn recreate_face(
    bm: &BMesh,
    verts: &[usize],
    edges: &[usize],
    target: &[usize],
) -> Result<Option<Compacted>, MeshError> {
    let n = verts.len();
    let mut out_verts = Vec::with_capacity(n);
    let mut out_edges = Vec::with_capacity(n);
    for (k, (&current, &edge)) in verts.iter().zip(edges).enumerate() {
        let next = verts[(k + 1) % n];
        let current_merged = target[current] != NONE;
        let next_merged = target[next] != NONE;
        let mapped_current = if current_merged {
            target[current]
        } else {
            current
        };
        let mapped_next = if next_merged { target[next] } else { next };
        let kept =
            if !current_merged && !next_merged {
                Some(edge)
            } else if mapped_current == mapped_next {
                None
            } else {
                Some(bm.edge_find(mapped_current, mapped_next).ok_or(
                    MeshError::InvalidTopology("uv sphere weld lost an edge between survivors"),
                )?)
            };
        if let Some(kept) = kept {
            if out_verts.contains(&mapped_current) {
                return Ok(None);
            }
            out_verts.push(mapped_current);
            out_edges.push(kept);
        }
    }
    if out_edges.len() < 3 {
        return Ok(None);
    }
    Ok(Some((out_verts, out_edges)))
}

/// Survivors in storage order with their rebuilt topology.
struct Topology {
    positions: Vec<DVec3>,
    edges: Vec<[usize; 2]>,
    faces: Vec<Vec<usize>>,
}

/// `remove_doubles` followed by `weld_verts` (`bmo_removedoubles.cc`) and the `BMesh` compaction.
fn weld(bm: &mut BMesh, duplicates: &[usize]) -> Result<Topology, MeshError> {
    let vertex_count = bm.co.len();
    let target: Vec<usize> = duplicates
        .iter()
        .enumerate()
        .map(|(index, &value)| if value == index { NONE } else { value })
        .collect();
    let map = |vertex: usize| {
        if target[vertex] == NONE {
            vertex
        } else {
            target[vertex]
        }
    };

    let edge_count = bm.edges.len();
    for edge in 0..edge_count {
        let [a, b] = bm.edges[edge].v;
        if target[a] == NONE && target[b] == NONE {
            continue;
        }
        let (v1, v2) = (map(a), map(b));
        if v1 == v2 {
            bm.edges[edge].collapsed = true;
        } else if bm.edge_find(v1, v2).is_none() {
            bm.edge_add(v1, v2);
        }
        bm.edges[edge].dead = true;
    }

    // remdoubles_splitface: a split would be needed if a merged corner shares a face with its
    // target at a non-adjacent corner. Never reached for a UV sphere; reject it rather than guess.
    for face in &bm.faces {
        let n = face.v.len();
        for (k, &vertex) in face.v.iter().enumerate() {
            if target[vertex] == NONE {
                continue;
            }
            for (k2, &other) in face.v.iter().enumerate() {
                let adjacent = (k2 + n - k) % n == 1 || (k + n - k2) % n == 1;
                if other == target[vertex] && k2 != k && !adjacent {
                    return Err(MeshError::InvalidTopology(
                        "uv sphere weld would split a face",
                    ));
                }
            }
        }
    }

    let face_count = bm.faces.len();
    for face_index in 0..face_count {
        let verts = bm.faces[face_index].v.clone();
        let edges = bm.faces[face_index].e.clone();
        if verts.iter().all(|&vertex| target[vertex] == NONE) {
            continue;
        }
        bm.faces[face_index].dead = true;
        let collapsed = edges
            .iter()
            .filter(|&&edge| bm.edges[edge].collapsed)
            .count();
        if verts.len() < collapsed + 3 {
            continue;
        }
        if let Some((new_verts, new_edges)) = recreate_face(bm, &verts, &edges, &target)? {
            // bmesh_face_swap_data keeps the old face slot; the temporary face is freed again.
            bm.faces[face_index].v = new_verts;
            bm.faces[face_index].e = new_edges;
            bm.faces[face_index].dead = false;
        }
    }

    let mut survivor = vec![NONE; vertex_count];
    let mut positions = Vec::new();
    for (vertex, &co) in bm.co.iter().enumerate() {
        if target[vertex] == NONE {
            survivor[vertex] = positions.len();
            positions.push(DVec3::new(
                f64::from(co[0]),
                f64::from(co[1]),
                f64::from(co[2]),
            ));
        }
    }
    let mut edges = Vec::new();
    for edge in bm.edges.iter().filter(|edge| !edge.dead) {
        let [a, b] = edge.v;
        if survivor[a] == NONE || survivor[b] == NONE {
            return Err(MeshError::InvalidTopology(
                "uv sphere edge references a merged vertex",
            ));
        }
        edges.push([survivor[a], survivor[b]]);
    }
    let mut faces = Vec::new();
    for face in bm.faces.iter().filter(|face| !face.dead) {
        let mut verts = Vec::with_capacity(face.v.len());
        for &vertex in &face.v {
            if survivor[vertex] == NONE {
                return Err(MeshError::InvalidTopology(
                    "uv sphere face references a merged vertex",
                ));
            }
            verts.push(survivor[vertex]);
        }
        faces.push(verts);
    }
    Ok(Topology {
        positions,
        edges,
        faces,
    })
}

/// Build the UV sphere in Blender's storage order.
///
/// # Errors
///
/// Returns an error when the parameters are invalid or the mesh IDs are exhausted.
pub(super) fn build(params: UvSphereParams) -> Result<Mesh, MeshError> {
    let segments = i32::try_from(params.segments).map_err(|_| MeshError::IdExhausted)?;
    let rings = i32::try_from(params.ring_count).map_err(|_| MeshError::IdExhausted)?;
    let radius = params.radius as f32;
    let topology = topology(segments, rings, radius)?;
    let mut mesh = Mesh::new();
    let mut ids = Vec::with_capacity(topology.positions.len());
    for position in topology.positions {
        ids.push(mesh.insert_vertex(position)?);
    }
    for [a, b] in topology.edges {
        let (id, next) = allocate(mesh.next_id.edge)?;
        mesh.edges.push(Edge {
            id,
            vertices: [ids[a], ids[b]],
        });
        mesh.next_id.edge = next;
    }
    for face in topology.faces {
        let (id, next) = allocate(mesh.next_id.face)?;
        mesh.faces.push(Face {
            id,
            vertices: face.iter().map(|&vertex| ids[vertex]).collect(),
            material_index: 0,
        });
        mesh.next_id.face = next;
    }
    Ok(mesh)
}

fn topology(segments: i32, rings: i32, radius: f32) -> Result<Topology, MeshError> {
    let mut seed = Vec::with_capacity(usize::try_from(rings + 1).unwrap_or(0));
    for ring in 0..=rings {
        let (sin, cos) = sin_cos_from_fraction(ring, 2 * rings);
        seed.push([0.0, radius * sin, radius * cos]);
    }
    let pivot = seed[rings as usize];
    let cmat = axis_angle_mat3(
        [0.0, 0.0, 1.0],
        (-(2.0 * std::f64::consts::PI) / f64::from(segments)) as f32,
    );
    let rotation = rotation_about_pivot(&cmat, pivot);

    let mut bm = BMesh::default();
    let seed_ids: Vec<usize> = seed.into_iter().map(|co| bm.vertex(co)).collect();
    let mut input: Vec<usize> = seed_ids
        .windows(2)
        .map(|pair| bm.edge_add(pair[0], pair[1]))
        .collect();
    for _ in 0..segments {
        input = extrude_edge_only(&mut bm, &input, &rotation);
    }

    // remove_doubles distance: a third of the shorter of the meridian and parallel segments.
    let phid = std::f32::consts::PI / rings as f32;
    let meridian = 2.0 * radius * (phid / 2.0).sin();
    let start = [radius * phid.sin(), 0.0, radius * phid.cos()];
    let rotated = mul_v3_m3v3(&cmat, start);
    let parallel = squared_distance(start, rotated).sqrt();
    let range = meridian.min(parallel) / 3.0;

    let duplicates = find_duplicates(&bm.co, range);
    weld(&mut bm, &duplicates)
}
