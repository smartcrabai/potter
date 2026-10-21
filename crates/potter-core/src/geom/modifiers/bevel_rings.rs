//! Blender's bevel topology (`bmesh_bevel.cc`, back end): the vertex mesh at each beveled vertex
//! (`build_vmesh` and its builders), the strips along beveled edges (`bevel_build_edge_polygons`),
//! and the rebuilt original faces (`bev_rebuild_polygon`), emitted as an [`Emission`].
//!
//! Each output corner names the source face whose UVs it interpolates, as Blender's
//! `bev_create_ngon` interpolates each loop from its face argument.

use std::collections::HashSet;

use glam::DVec3;

use super::Polygon;
use super::bevel_boundary::{
    EdgeHalf, Front, MeshKind, PRO_LINE_R, PRO_SQUARE_IN_R, PRO_SQUARE_R, internal_invariant,
};
use super::bevel_shading::Kind;
use super::bevel_vmesh::{
    EdgeState, Emission, Profile, ProfileSpacing, VMesh, bounds_center, closest_on_segment,
    cube_corner_vmesh, general_vmesh, pipe_adj_vmesh, point_strictly_inside, profile_fullness,
    project_to_face, square_out_adj_vmesh, two_edge_vertex_samples, weld_profile_samples,
};
use crate::error::Result;
use crate::geom::edge_key;

/// Blender's `BEVEL_EPSILON`.
const EPSILON: f64 = 1.0e-6;
/// Blender's `BEVEL_EPSILON_BIG`.
const EPSILON_BIG: f64 = 1.0e-4;

/// Per-bevel facts beyond the front end that the topology reads.
pub(super) struct Context {
    /// Connected UV-island label of each face (Blender's `math_layer_info.face_component`).
    pub components: Option<Vec<usize>>,
    /// The mesh has UV layers (Blender's `has_math_layers`).
    pub has_math_layers: bool,
    /// The bevel's material override (`material >= 0`).
    pub material: Option<u32>,
}

/// One beveled vertex's vertex mesh. Each storage index names the cell whose output point it uses
/// (Blender's shared `BMVert`): canonical cells of an ADJ mesh, and the copies Blender makes on
/// profile grids. Positions Blender never creates are undefined; each output point is created on
/// first use.
struct Grid {
    vm: VMesh,
    defined: Vec<bool>,
    representative: Vec<usize>,
    ids: Vec<Option<usize>>,
}

impl Grid {
    fn new(vm: VMesh, canonical: bool) -> Self {
        let len = vm.co.len();
        let representative = if canonical {
            vm.canonical_representatives()
        } else {
            (0..len).collect()
        };
        Self {
            vm,
            defined: vec![true; len],
            representative,
            ids: vec![None; len],
        }
    }

    /// The representative storage slot of grid position `(i, j, k)`, when that position is defined.
    fn slot(&self, i: usize, j: usize, k: usize) -> Option<usize> {
        let index = self.representative[self.vm.index(i, j, k)];
        self.defined[index].then_some(index)
    }
}

/// A polygon corner before emission: output point, snapped edge endpoints, and UV source face.
type Corner = (usize, Option<(DVec3, DVec3)>, Option<usize>);

/// Builds the output of one bevel from its front end.
pub(super) struct Builder<'a> {
    front: &'a Front,
    polygons: &'a [Polygon],
    points: &'a [DVec3],
    context: &'a Context,
    /// Per input vertex: its vertex mesh, once built.
    grids: Vec<Option<Grid>>,
    /// Effective mesh kind after Blender's weld and pipe overrides in `build_vmesh`.
    mesh_kinds: Vec<Option<MeshKind>>,
    /// Square-in tri-corners use profile-only cells and a center polygon, not ADJ quads.
    square_in: Vec<bool>,
    /// Blender's `bndv->index`: each boundary vertex's position in its vertex's ring.
    positions: Vec<usize>,
    emission: Emission,
    /// Source vertices touched by an input face, computed once during construction.
    vertex_has_face: Vec<bool>,
    wire_edges: Vec<Vec<(usize, usize)>>,
    /// Wire edges the topology reattaches or creates.
    loose: Vec<[usize; 2]>,
}

/// Blender's `boundvert_rep_face`: the face whose UVs interpolate boundary vertex `b`, and a second
/// candidate when the boundary's faces disagree.
fn boundvert_rep_face(front: &Front, b: usize) -> Option<usize> {
    let bound = &front.bounds[b];
    let edges: &[EdgeHalf] = &front.edges;
    if let Some(ebev) = bound.ebev {
        return edges[ebev].fprev;
    }
    if let Some(efirst) = bound.efirst {
        let first = &edges[efirst];
        let last = bound.elast.map(|id| &edges[id]);
        if let Some(face) = first.fprev {
            return Some(face);
        }
        if let Some(face) = first.fnext {
            return Some(face);
        }
        return last.and_then(|last| last.fprev);
    }
    let prev = &front.bounds[bound.prev];
    prev.elast?;
    let prev_elast = prev.elast.and_then(|id| edges[id].fnext);
    prev_elast.or_else(|| {
        front.bounds[bound.next]
            .efirst
            .and_then(|id| edges[id].fprev)
    })
}

/// Blender's `choose_rep_face`: of `faces`, the one that wins the tie-breaks in order: the math-layer
/// component, the material index, then the bounding-box center's z, x, and y. Blender's selection
/// tie-break is omitted because the bevel input carries no face selection.
fn choose_rep_face(
    faces: &[Option<usize>],
    context: &Context,
    polygons: &[Polygon],
    points: &[DVec3],
) -> Option<usize> {
    let count = faces.len();
    let mut values = vec![[0.0_f64; 5]; count];
    let mut still_viable = vec![false; count];
    let mut num_viable = 0;
    for (f, face) in faces.iter().enumerate() {
        let Some(index) = *face else {
            continue;
        };
        still_viable[f] = true;
        num_viable += 1;
        let center = bounds_center(points, &polygons[index].0);
        values[f] = [
            context
                .components
                .as_ref()
                .map_or(0.0, |labels| labels[index] as f64),
            f64::from(polygons[index].1),
            center.z,
            center.x,
            center.y,
        ];
    }
    let mut best: Option<usize> = None;
    for value_index in [0_usize, 1, 2, 3, 4] {
        if num_viable <= 1 {
            break;
        }
        for f in 0..count {
            if !still_viable[f] || best == Some(f) {
                continue;
            }
            let Some(current) = best else {
                best = Some(f);
                continue;
            };
            if values[f][value_index] < values[current][value_index] {
                best = Some(f);
                for i in (0..f).rev() {
                    if still_viable[i] {
                        still_viable[i] = false;
                        num_viable -= 1;
                    }
                }
            } else if values[f][value_index] > values[current][value_index] {
                still_viable[f] = false;
                num_viable -= 1;
            }
        }
    }
    faces[best.unwrap_or(0)]
}

/// Blender's `get_incident_edges`: the other endpoints of the first two edges of `face` that touch
/// `vertex`, in face order.
fn incident_neighbors(polygons: &[Polygon], face: usize, vertex: usize) -> Option<(usize, usize)> {
    let vertices = &polygons[face].0;
    let count = vertices.len();
    let mut neighbors = Vec::with_capacity(2);
    for i in 0..count {
        let (a, b) = (vertices[i], vertices[(i + 1) % count]);
        if a == vertex {
            neighbors.push(b);
        } else if b == vertex {
            neighbors.push(a);
        }
    }
    match neighbors.as_slice() {
        [first, second, ..] => Some((*first, *second)),
        _ => None,
    }
}

/// Blender's `find_closer_edge`: of the edges `vertex`-`first` and `vertex`-`second`, the one whose
/// segment is nearer `co`, as its endpoint pair.
fn closer_edge(
    points: &[DVec3],
    co: DVec3,
    vertex: usize,
    first: usize,
    second: usize,
) -> (DVec3, DVec3) {
    let distance = |other: usize| {
        let (a, b) = (points[vertex], points[other]);
        closest_on_segment(co, a, b).distance_squared(co)
    };
    let other = if distance(first) < distance(second) {
        first
    } else {
        second
    };
    (points[vertex], points[other])
}

/// Blender's `snap_edge_for_center_vmesh_vert`.
fn snap_edge_for_center(
    i: usize,
    n: usize,
    eprev: Option<usize>,
    enext: Option<usize>,
    reps: &[Option<usize>],
    center_frep: Option<usize>,
    beats: &[bool],
) -> Option<usize> {
    let previ = (i + n - 1) % n;
    let nexti = (i + 1) % n;
    if beats[previ] && reps[previ] == center_frep {
        return eprev;
    }
    if !beats[i] && reps[nexti] == center_frep {
        return enext;
    }
    None
}

/// Blender's `snap_edges_for_vmesh_vert`: the edges (`b1`, `b2`, `b3` are the seam edges before,
/// at, and after boundary vertex `i`) that the four corners of quad `(i, j, k)` snap to, for odd
/// segment counts.
fn snap_edges_for_vmesh_vert(
    i: usize,
    j: usize,
    k: usize,
    ns2: usize,
    n: usize,
    b1: Option<usize>,
    b2: Option<usize>,
    b3: Option<usize>,
    reps: &[Option<usize>],
    center_frep: Option<usize>,
    beats: &[bool],
) -> [Option<usize>; 4] {
    let mut snaps = [None; 4];
    let previ = (i + n - 1) % n;
    for (corner, snap) in snaps.iter_mut().enumerate() {
        let jj = if corner < 2 { j } else { j + 1 };
        let kk = if corner == 0 || corner == 3 { k } else { k + 1 };
        *snap = if jj < ns2 && kk < ns2 {
            None
        } else if jj < ns2 && kk == ns2 {
            // The left side of the center strip, not the center polygon.
            (!beats[i]).then_some(b2).flatten()
        } else if jj < ns2 && kk == ns2 + 1 {
            // The right side of the center strip, not the center polygon.
            beats[i].then_some(b2).flatten()
        } else if jj == ns2 && kk < ns2 {
            // The top of the top strip, not the center polygon.
            beats[previ].then_some(b1).flatten()
        } else if jj == ns2 && kk == ns2 {
            // The center polygon's vertex for boundary vertex i.
            snap_edge_for_center(i, n, b1, b2, reps, center_frep, beats)
        } else if jj == ns2 && kk == ns2 + 1 {
            // The center polygon's vertex for boundary vertex i + 1.
            snap_edge_for_center((i + 1) % n, n, b2, b3, reps, center_frep, beats)
        } else {
            None
        };
    }
    snaps
}

impl<'a> Builder<'a> {
    /// Blender's `bevel_vert_two_edges`: a vertex bevel with two boundary vertices has no vertex
    /// polygon; its profile is the only topology, and without faces at the vertex that profile chain
    /// is made of wire edges.
    fn emit_two_edge(&mut self, vertex: usize) -> Result<()> {
        if !self.front.p.affect_vertices || self.vertex_has_face[vertex] {
            return Ok(());
        }
        let seg = self.front.p.seg;
        let mut previous = self
            .id(vertex, 0, 0, 0)?
            .ok_or_else(|| internal_invariant("two-edge profile start defined"))?;
        for k in 1..=seg {
            let current = self
                .id(vertex, 0, 0, k)?
                .ok_or_else(|| internal_invariant("two-edge profile point defined"))?;
            self.loose.push([previous, current]);
            previous = current;
        }
        Ok(())
    }

    fn new(
        front: &'a Front,
        points: &'a [DVec3],
        polygons: &'a [Polygon],
        context: &'a Context,
    ) -> Self {
        let mut positions = vec![0; front.bounds.len()];
        for bev in front.verts.iter().flatten() {
            for (index, id) in bev.ring.iter().enumerate() {
                positions[*id] = index;
            }
        }
        let mut vertex_has_face = vec![false; front.verts.len()];
        let mut face_edges = HashSet::new();
        for (face, _) in polygons {
            for index in 0..face.len() {
                let first = face[index];
                let second = face[(index + 1) % face.len()];
                vertex_has_face[first] = true;
                if !front.p.affect_vertices {
                    face_edges.insert(edge_key(first, second));
                }
            }
        }
        let wire_edges = if front.p.affect_vertices {
            Vec::new()
        } else {
            let mut wire_edges = vec![Vec::new(); front.verts.len()];
            for &(first, second) in &front.geo.edge_order {
                if !face_edges.contains(&edge_key(first, second)) {
                    wire_edges[first].push((first, second));
                    wire_edges[second].push((first, second));
                }
            }
            wire_edges
        };
        Self {
            front,
            polygons,
            points,
            context,
            grids: (0..front.verts.len()).map(|_| None).collect(),
            mesh_kinds: vec![None; front.verts.len()],
            square_in: vec![false; front.verts.len()],
            positions,
            emission: Emission::new(points),
            vertex_has_face,
            wire_edges,
            loose: Vec::new(),
        }
    }

    fn bev(&self, vertex: usize) -> Result<&super::bevel_boundary::BevVert> {
        self.front.verts[vertex]
            .as_ref()
            .ok_or_else(|| internal_invariant("beveled vertex"))
    }

    fn grid(&self, vertex: usize) -> Result<&Grid> {
        self.grids[vertex]
            .as_ref()
            .ok_or_else(|| internal_invariant("vertex mesh built"))
    }

    fn grid_mut(&mut self, vertex: usize) -> Result<&mut Grid> {
        self.grids[vertex]
            .as_mut()
            .ok_or_else(|| internal_invariant("vertex mesh built"))
    }

    /// The output point of grid position `(i, j, k)` of `vertex`, created on first use.
    fn id(&mut self, vertex: usize, i: usize, j: usize, k: usize) -> Result<Option<usize>> {
        let Some(index) = self.grid(vertex)?.slot(i, j, k) else {
            return Ok(None);
        };
        if let Some(id) = self.grid(vertex)?.ids[index] {
            return Ok(Some(id));
        }
        let co = self.grid(vertex)?.vm.co[index];
        let id = self.emission.push_point(co, vertex);
        self.grid_mut(vertex)?.ids[index] = Some(id);
        Ok(Some(id))
    }

    /// The coordinates of a defined grid position.
    fn grid_point(&self, vertex: usize, i: usize, j: usize, k: usize) -> Result<Option<DVec3>> {
        let grid = self.grid(vertex)?;
        Ok(grid.slot(i, j, k).map(|index| grid.vm.co[index]))
    }

    /// The boundary position `i` of a boundary vertex in its vertex's ring.
    fn position(&self, bound: usize) -> usize {
        self.positions[bound]
    }

    /// The endpoints of an edge half's edge.
    fn edge_points(&self, edge: usize) -> (DVec3, DVec3) {
        let e = &self.front.edges[edge];
        (self.points[e.vert], self.points[e.other])
    }

    /// The edge half of `vertex` on the undirected edge `key`.
    fn edge_half(&self, vertex: usize, key: (usize, usize)) -> Option<usize> {
        self.front.verts[vertex]
            .as_ref()?
            .edges
            .iter()
            .copied()
            .find(|id| {
                let e = &self.front.edges[*id];
                edge_key(e.vert, e.other) == key
            })
    }

    /// The face used when a corner has no representative face: the first face at the vertex.
    fn fallback_face(&self, vertex: usize) -> Result<usize> {
        let e = &self.front.edges[self.bev(vertex)?.edges[0]];
        Ok(e.fprev.or(e.fnext).unwrap_or(0))
    }

    fn emit_polygon(
        &mut self,
        vertices: Vec<usize>,
        corners: Vec<Option<(DVec3, DVec3)>>,
        sources: Vec<Option<usize>>,
        face: usize,
        kind: Kind,
    ) {
        self.emission.push_polygon(
            vertices,
            corners,
            sources,
            face,
            kind,
            self.context.material,
        );
    }

    /// Blender's `build_vmesh` before its kind dispatch: the vertex mesh of `vertex`.
    fn build_grid(&mut self, vertex: usize) -> Result<()> {
        let (count, selcount, front_kind) = {
            let bev = self.bev(vertex)?;
            (bev.ring.len(), bev.selcount, bev.mesh_kind)
        };
        let weld = selcount == 2 && count == 2;
        if weld {
            let grid = self.profile_grid(vertex, true)?;
            self.grids[vertex] = Some(grid);
            self.mesh_kinds[vertex] = Some(MeshKind::None);
            return Ok(());
        }

        let pipe = self.pipe_test(vertex)?;
        let kind = if pipe.is_some() {
            MeshKind::Adj
        } else {
            front_kind
        };
        self.mesh_kinds[vertex] = Some(kind);
        if kind != MeshKind::Adj {
            let grid = self.profile_grid(vertex, false)?;
            self.grids[vertex] = Some(grid);
            return Ok(());
        }

        let square_out = self.front.p.pro_super_r == PRO_SQUARE_R
            && selcount >= 3
            && self.front.p.seg.is_multiple_of(2)
            && self.front.p.custom.is_none();
        let square_in = pipe.is_none()
            && self.front.p.pro_super_r == PRO_SQUARE_IN_R
            && self.tri_corner_test(vertex)? == 1
            && self.front.p.custom.is_none();
        let vm = if square_out {
            square_out_adj_vmesh(self.front, vertex)?
        } else if square_in {
            let ring = &self.bev(vertex)?.ring;
            let spacing = ProfileSpacing::new(self.front.p.seg, self.front.p.pro_super_r);
            cube_corner_vmesh(
                &spacing,
                [
                    self.front.bounds[ring[0]].co,
                    self.front.bounds[ring[1]].co,
                    self.front.bounds[ring[2]].co,
                ],
                self.points[vertex],
            )
        } else {
            let vm = self.adj_vmesh(vertex)?;
            if let Some(vpipe) = pipe {
                pipe_adj_vmesh(vm, self.front, vertex, vpipe)?
            } else {
                vm
            }
        };
        let mut grid = Grid::new(vm, !square_in);
        if square_in {
            self.square_in[vertex] = true;
            self.share_square_in_cells(vertex, &mut grid)?;
        }
        self.grids[vertex] = Some(grid);
        Ok(())
    }

    /// Blender's `pipe_test`: identifies a straight beveled three-edge sequence whose incident
    /// face normals are perpendicular to the pipe direction.
    fn pipe_test(&self, vertex: usize) -> Result<Option<usize>> {
        let bev = self.bev(vertex)?;
        let count = bev.ring.len();
        if !(3..=4).contains(&count) || !(3..=4).contains(&bev.selcount) || self.front.p.seg <= 1 {
            return Ok(None);
        }
        let point = self.points[vertex].as_vec3();
        for index in 0..count {
            let next = (index + 1) % count;
            let next_next = (index + 2) % count;
            let (Some(first), Some(middle), Some(last)) = (
                self.front.bounds[bev.ring[index]].ebev,
                self.front.bounds[bev.ring[next]].ebev,
                self.front.bounds[bev.ring[next_next]].ebev,
            ) else {
                continue;
            };
            if !self.front.edges[first].is_bev
                || !self.front.edges[middle].is_bev
                || !self.front.edges[last].is_bev
            {
                continue;
            }
            let dir1 =
                (point - self.points[self.front.edges[first].other].as_vec3()).normalize_or_zero();
            let dir3 =
                (self.points[self.front.edges[last].other].as_vec3() - point).normalize_or_zero();
            if dir1.length_squared() == 0.0 || dir3.length_squared() == 0.0 {
                continue;
            }
            if dir1.dot(dir3).clamp(-1.0, 1.0).acos() >= 2.0_f32 * std::f32::consts::PI / 180.0_f32
            {
                continue;
            }
            if bev.edges.iter().any(|id| {
                self.front.edges[*id].fnext.is_some_and(|face| {
                    dir1.dot(self.front.geo.face_normals[face].as_vec3()).abs() > EPSILON_BIG as f32
                })
            }) {
                continue;
            }
            return Ok(Some(bev.ring[index]));
        }
        Ok(None)
    }

    /// Blender's `build_square_in_vmesh`: only profile cells remain, with the inward-square
    /// boundary sharing pattern and, for odd segment counts, the center polygon cells.
    fn share_square_in_cells(&self, vertex: usize, grid: &mut Grid) -> Result<()> {
        let count = self.bev(vertex)?.ring.len();
        let seg = self.front.p.seg;
        let half = seg / 2;
        for i in 0..count {
            let end = grid.vm.index(i, 0, seg);
            let next_start = grid.vm.index((i + 1) % count, 0, 0);
            grid.representative[end] = grid.representative[next_start];
            for k in 1..seg {
                let source = if i > 0 && k <= half {
                    Some(grid.vm.index(i - 1, 0, seg - k))
                } else if i + 1 == count && k > half {
                    Some(grid.vm.index(0, 0, seg - k))
                } else {
                    None
                };
                let target = grid.vm.index(i, 0, k);
                if let Some(source) = source {
                    grid.representative[target] = grid.representative[source];
                }
            }
        }
        if seg % 2 == 1 {
            for i in 0..count {
                let center = grid.vm.index(i, half, half);
                let profile = grid.vm.index(i, 0, half);
                grid.representative[center] = grid.representative[profile];
            }
        }
        Ok(())
    }

    /// The boundary and profile points of `vertex`'s ring, as Blender's `build_vmesh` lays them out
    /// before the kind-specific mesh is built.
    fn profile_grid(&self, vertex: usize, weld: bool) -> Result<Grid> {
        let ring = &self.bev(vertex)?.ring;
        let count = ring.len();
        let seg = self.front.p.seg;
        let mut vm = VMesh::new(count, seg);
        let mut defined = vec![false; vm.co.len()];
        // Blender's `copy_mesh_vert`: a cell that copies another shares its output point.
        let mut copies: Vec<(usize, usize)> = Vec::new();
        for (i, id) in ring.iter().enumerate() {
            let index = vm.index(i, 0, 0);
            vm.co[index] = self.front.bounds[*id].co;
            defined[index] = true;
            let index = vm.index(i, 0, seg);
            vm.co[index] = self.front.bounds[self.front.bounds[*id].next].co;
            defined[index] = true;
            copies.push((index, vm.index((i + 1) % count, 0, 0)));
        }
        for (i, id) in ring.iter().enumerate() {
            let bound = &self.front.bounds[*id];
            if bound.ebev.is_some() {
                for k in 1..seg {
                    let index = vm.index(i, 0, k);
                    vm.co[index] = bound.profile.prof_co[k];
                    defined[index] = true;
                }
            } else if count == 2 {
                // One beveled edge: the other boundary vertex's profile runs back to this one.
                for k in 1..seg {
                    let source = vm.index(1 - i, 0, seg - k);
                    let index = vm.index(i, 0, k);
                    vm.co[index] = vm.co[source];
                    defined[index] = defined[source];
                    copies.push((index, source));
                }
            }
        }
        if self.front.p.affect_vertices && count == 2 && seg > 1 {
            // Blender's two-edge vertex bevel recalculates one projection-free profile between the
            // boundary points and mirrors its samples onto the second boundary.
            let profile = two_edge_vertex_samples(self.front, vertex)
                .ok_or_else(|| internal_invariant("two-edge vertex profile is available"))?;
            for k in 1..seg {
                let sample = *profile
                    .get(k)
                    .ok_or_else(|| internal_invariant("two-edge vertex profile sample defined"))?;
                let index = vm.index(0, 0, k);
                vm.co[index] = sample;
                defined[index] = true;
                let mirror = vm.index(1, 0, seg - k);
                vm.co[mirror] = sample;
                defined[mirror] = true;
                copies.push((mirror, index));
            }
        }
        let mut grid = Grid::new(vm, false);
        grid.defined = defined;
        for (to, from) in copies {
            grid.representative[to] = grid.representative[from];
        }
        if weld && !self.front.p.affect_vertices {
            self.share_weld_profiles(vertex, &mut grid);
        }
        Ok(grid)
    }

    /// Blender's two-boundary weld uses one profile chain at both ends of the shared boundary.
    fn share_weld_profiles(&self, vertex: usize, grid: &mut Grid) {
        let Some((first, second, samples1, samples2, super_r1, super_r2)) =
            weld_profile_samples(self.front, vertex)
        else {
            return;
        };
        let (i1, i2) = (self.position(first), self.position(second));
        let seg = self.front.p.seg;
        let line1 = super_r1 == PRO_LINE_R;
        let line2 = super_r2 == PRO_LINE_R;
        for k in 1..seg {
            let first_index = grid.vm.index(i1, 0, k);
            let second_index = grid.vm.index(i2, 0, seg - k);
            let co1 = samples1[k];
            let co2 = samples2[seg - k];
            let co = if self.front.p.custom.is_none() && line1 && !line2 {
                co2
            } else if self.front.p.custom.is_none() && line2 && !line1 {
                co1
            } else {
                ((co1.as_vec3() + co2.as_vec3()) * 0.5).as_dvec3()
            };
            grid.vm.co[first_index] = co;
            grid.vm.co[second_index] = co;
            grid.representative[second_index] = grid.representative[first_index];
        }
    }

    /// Blender's `adj_vmesh`: the three-edge tri-corner mesh when `tri_corner_test` allows it,
    /// otherwise the general ADJ mesh subdivided from its boundary profiles.
    fn adj_vmesh(&self, vertex: usize) -> Result<VMesh> {
        let seg = self.front.p.seg;
        let bounds: Vec<DVec3> = self
            .bev(vertex)?
            .ring
            .iter()
            .map(|id| self.front.bounds[*id].co)
            .collect();
        if bounds.len() == 3
            && self.tri_corner_test(vertex)? != -1
            && self.front.p.pro_super_r != PRO_SQUARE_IN_R
        {
            let spacing = ProfileSpacing::new(seg, self.front.p.pro_super_r);
            return Ok(cube_corner_vmesh(
                &spacing,
                [bounds[0], bounds[1], bounds[2]],
                self.points[vertex],
            ));
        }
        let seg_2 = self.front.pro_spacing.seg_2;
        let profiles: Vec<Profile> = self
            .bev(vertex)?
            .ring
            .iter()
            .map(|id| Profile::from_bound(&self.front.bounds[*id].profile, seg, seg_2))
            .collect();
        let fullness = if self.front.p.custom.is_some() {
            let denominator = 2.0 * seg as f64;
            self.front
                .pro_spacing
                .xvals
                .iter()
                .zip(&self.front.pro_spacing.yvals)
                .take(seg)
                .map(|(x, y)| (x + y) / denominator)
                .sum()
        } else {
            profile_fullness(seg, self.front.p.profile, self.front.p.pro_super_r)
        };
        Ok(general_vmesh(
            seg,
            fullness,
            self.front.p.custom.is_some(),
            &profiles,
            self.points[vertex],
        ))
    }

    /// Blender's `BM_edge_calc_face_angle_signed_ex`, including the convexity sign convention.
    fn signed_face_angle(&self, id: usize) -> f32 {
        let edge = &self.front.edges[id];
        // The newer BMesh radial loop is the signed-angle reference face.
        let [second, first] = self.front.edge_faces(edge.edge_id) else {
            return 0.0;
        };
        let n1 = self.front.geo.face_normals[*first].as_vec3();
        let n2 = self.front.geo.face_normals[*second].as_vec3();
        let angle = n1.dot(n2).clamp(-1.0, 1.0).acos();
        let ring = &self.polygons[*first].0;
        for index in 0..ring.len() {
            let a = ring[index];
            let b = ring[(index + 1) % ring.len()];
            if edge_key(a, b) == edge_key(edge.vert, edge.other) {
                let direction = (self.points[b] - self.points[a]).as_vec3();
                let convex = direction.dot(n1.cross(n2)) > 0.0;
                return if convex { angle } else { -angle };
            }
        }
        0.0
    }

    /// Blender's `tri_corner_test`: 1 when a three-edge corner is a cube corner (every edge beveled,
    /// right dihedral angles, equal offsets), 0 when the tri-corner mesh applies without that shape,
    /// and -1 when it does not apply.
    fn tri_corner_test(&self, vertex: usize) -> Result<i32> {
        if self.front.p.affect_vertices || self.front.p.custom.is_some() {
            return Ok(-1);
        }
        let bev = self.bev(vertex)?;
        if bev.ring.len() != 3 {
            return Ok(0);
        }
        let offset = self.front.edges[bev.edges[0]].offset_l as f32;
        let mut in_plane_e = 0_usize;
        let mut totang = 0.0_f32;
        for id in &bev.edges {
            let edge = &self.front.edges[*id];
            let ang = self.signed_face_angle(*id);
            let absang = ang.abs();
            if absang <= std::f32::consts::FRAC_PI_4 {
                in_plane_e += 1;
            } else if absang >= 3.0_f32 * std::f32::consts::FRAC_PI_4 {
                return Ok(-1);
            }
            if edge.is_bev && (edge.offset_l as f32 - offset).abs() > EPSILON as f32 {
                return Ok(-1);
            }
            totang += ang;
        }
        if in_plane_e != bev.edges.len() - 3 {
            return Ok(-1);
        }
        let angdiff = (totang.abs() - 3.0_f32 * std::f32::consts::FRAC_PI_2).abs();
        if (self.front.p.pro_super_r == PRO_SQUARE_R && angdiff > std::f32::consts::PI / 16.0_f32)
            || angdiff > std::f32::consts::FRAC_PI_4
        {
            return Ok(-1);
        }
        if bev.edges.len() != 3 || bev.selcount != 3 {
            return Ok(0);
        }
        Ok(1)
    }

    /// Blender's `frep_for_center_poly`: the face whose UVs interpolate the center of `vertex`'s
    /// vertex mesh, chosen among the representative faces of the vertex's edges.
    fn frep_for_center_poly(&self, vertex: usize) -> Result<Option<usize>> {
        let bev = self.bev(vertex)?;
        let consider_all =
            bev.selcount == 1 || (self.front.p.affect_vertices && self.front.p.seg % 2 == 1);
        let mut fchoices: Vec<usize> = Vec::new();
        let mut any = None;
        for id in &bev.edges {
            let edge = &self.front.edges[*id];
            if !edge.is_bev && !consider_all {
                continue;
            }
            let Some(face) = choose_rep_face(
                &[edge.fprev, edge.fnext],
                self.context,
                self.polygons,
                self.points,
            ) else {
                continue;
            };
            any.get_or_insert(face);
            if fchoices.contains(&face) {
                continue;
            }
            // A face whose UV polygon would have no area cannot represent the center.
            if self.context.has_math_layers
                && self.projected_boundary_area(vertex, face)? < EPSILON_BIG
            {
                continue;
            }
            fchoices.push(face);
        }
        if fchoices.is_empty() {
            return Ok(any);
        }
        let options: Vec<Option<usize>> = fchoices.into_iter().map(Some).collect();
        Ok(choose_rep_face(
            &options,
            self.context,
            self.polygons,
            self.points,
        ))
    }

    /// Blender's `find_face_internal_boundverts`: up to three boundary vertices of `vertex` inside
    /// or on `face`, in ring order.
    fn face_internal_boundverts(&self, vertex: usize, face: usize) -> Result<Vec<usize>> {
        let normal = self.front.geo.face_normals[face];
        let face_points = &self.polygons[face].0;
        let mut found = Vec::new();
        for id in &self.bev(vertex)?.ring {
            if point_strictly_inside(self.points, face_points, normal, self.front.bounds[*id].co) {
                found.push(*id);
                if found.len() == 3 {
                    break;
                }
            }
        }
        Ok(found)
    }

    /// Blender's `projected_boundary_area`: the area of `vertex`'s boundary seen from `face`, with
    /// each boundary point outside the face snapped to the nearer of the face's two edges there.
    fn projected_boundary_area(&self, vertex: usize, face: usize) -> Result<f64> {
        let Some((n1, n2)) = incident_neighbors(self.polygons, face, vertex) else {
            return Ok(0.0);
        };
        let normal = self.front.geo.face_normals[face];
        let internal = self.face_internal_boundverts(vertex, face)?;
        let v = self.points[vertex];
        let projected: Vec<(f64, f64)> = self
            .bev(vertex)?
            .ring
            .iter()
            .map(|id| {
                let co = self.front.bounds[*id].co;
                if internal.contains(id) {
                    return project_to_face(normal, co);
                }
                let snap1 = closest_on_segment(co, v, self.points[n1]);
                let snap2 = closest_on_segment(co, v, self.points[n2]);
                let snap = if (snap1 - co).length_squared() <= (snap2 - co).length_squared() {
                    snap1
                } else {
                    snap2
                };
                project_to_face(normal, snap)
            })
            .collect();
        let count = projected.len();
        let twice_area: f64 = (0..count)
            .map(|i| {
                let (x0, y0) = projected[i];
                let (x1, y1) = projected[(i + 1) % count];
                x0 * y1 - x1 * y0
            })
            .sum();
        Ok(0.5 * twice_area.abs())
    }

    /// Blender's `bevel_build_rings` for an ADJ vertex mesh on edge bevels: the quads of each
    /// boundary strip, and for odd segment counts the center polygon.
    fn emit_adj(&mut self, vertex: usize) -> Result<()> {
        let affect_vertices = self.front.p.affect_vertices;
        if self.square_in[vertex] {
            if self.front.p.seg % 2 == 1 {
                let half = self.front.p.seg / 2;
                self.emit_center_ngon(vertex, half, half)?;
            }
            return Ok(());
        }
        let ring = self.bev(vertex)?.ring.clone();
        let count = ring.len();
        let seg = self.front.p.seg;
        let ns2 = seg / 2;
        let odd = seg % 2 == 1;
        let any_seam = self.bev(vertex)?.any_seam;
        let reps: Vec<Option<usize>> = ring
            .iter()
            .map(|id| boundvert_rep_face(self.front, *id))
            .collect();
        let edge_center = odd && !affect_vertices;
        let mut center_frep = None;
        let mut frep_beats_next = Vec::new();
        if edge_center {
            center_frep = self.frep_for_center_poly(vertex)?;
            frep_beats_next = vec![false; count];
            for (i, beats) in frep_beats_next.iter_mut().enumerate() {
                let inext = (i + 1) % count;
                let winner = choose_rep_face(
                    &[reps[i], reps[inext]],
                    self.context,
                    self.polygons,
                    self.points,
                );
                *beats = winner == reps[i];
            }
        }
        let mut center_verts = if edge_center {
            vec![None; count]
        } else {
            Vec::new()
        };
        let mut center_snaps = if edge_center {
            vec![None; count]
        } else {
            Vec::new()
        };
        let mut center_interps = if edge_center {
            vec![None; count]
        } else {
            Vec::new()
        };
        for (i, bound) in ring.iter().enumerate() {
            let inext = (i + 1) % count;
            let iprev = (i + count - 1) % count;
            let f = reps[i];
            let f2 = reps[inext];
            let fc = if frep_beats_next.get(i).copied().unwrap_or(false) {
                f
            } else {
                f2
            };
            // Vertex bevels take their edges from the boundary's first edge half, not its bevel edge.
            let edge_of = |b: usize| {
                if affect_vertices {
                    self.front.bounds[b].efirst
                } else {
                    self.front.bounds[b].ebev
                }
            };
            let ebev = edge_of(*bound);
            let eprev = edge_of(ring[iprev]);
            let enext = edge_of(ring[inext]);
            let front = self.front;
            let seam = |edge: Option<usize>| edge.filter(|id| front.edges[*id].is_seam);
            for j in 0..ns2 {
                for k in 0..(ns2 + usize::from(odd)) {
                    let ids = [
                        self.id(vertex, i, j, k)?,
                        self.id(vertex, i, j, k + 1)?,
                        self.id(vertex, i, j + 1, k + 1)?,
                        self.id(vertex, i, j + 1, k)?,
                    ];
                    let mut fr = [f; 4];
                    let mut se: [Option<usize>; 4] = [None; 4];
                    if affect_vertices {
                        // Blender's vertex-only rules: every corner interpolates from the next
                        // boundary's representative face, except the seam-free fan corner.
                        fr = [f2; 4];
                        if j < k {
                            if k == ns2 && j + 1 == ns2 {
                                se[2] = self.front.bounds[self.front.bounds[*bound].next].efirst;
                                se[3] = ebev;
                            }
                        } else if j == k {
                            se[0] = ebev;
                            se[2] = ebev;
                            if !ebev.is_some_and(|id| self.front.edges[id].is_seam) {
                                fr[3] = f;
                            }
                        }
                    } else if odd {
                        se = snap_edges_for_vmesh_vert(
                            i,
                            j,
                            k,
                            ns2,
                            count,
                            seam(eprev),
                            seam(ebev),
                            seam(enext),
                            &reps,
                            center_frep,
                            &frep_beats_next,
                        );
                        if k == ns2 {
                            let on_seam = ebev.is_none_or(|id| self.front.edges[id].is_seam);
                            fr = if on_seam { [fc; 4] } else { [f, f2, f2, f] };
                            if j == ns2 - 1 {
                                center_verts[i] = ids[3];
                                center_snaps[i] = se[3];
                                center_interps[i] = if any_seam { center_frep } else { f };
                            }
                        }
                    } else {
                        if k == ns2 - 1 {
                            se[1] = ebev;
                        }
                        if j == ns2 - 1 && eprev.is_some() {
                            se[3] = eprev;
                        }
                        se[2] = se[1].or(se[3]);
                    }
                    let Some(vertices) = ids.into_iter().collect::<Option<Vec<usize>>>() else {
                        continue;
                    };
                    let corners = se
                        .iter()
                        .map(|edge| edge.map(|id| self.edge_points(id)))
                        .collect();
                    let face = match fr[0] {
                        Some(face) => face,
                        None => self.fallback_face(vertex)?,
                    };
                    self.emit_polygon(vertices, corners, fr.to_vec(), face, Kind::Corner);
                }
            }
        }
        if odd {
            if affect_vertices {
                self.emit_center_ngon(vertex, ns2, ns2)?;
                return Ok(());
            }
            let Some(vertices) = center_verts.iter().copied().collect::<Option<Vec<usize>>>()
            else {
                return Ok(());
            };
            let corners = center_snaps
                .iter()
                .map(|edge| edge.map(|id| self.edge_points(id)))
                .collect();
            let face = match (if any_seam { center_frep } else { None }).or(center_interps[0]) {
                Some(face) => face,
                None => self.fallback_face(vertex)?,
            };
            self.emit_polygon(vertices, corners, center_interps, face, Kind::Corner);
        }
        Ok(())
    }

    /// Blender's `build_center_ngon`, used for odd vertex meshes and inward-square tri-corners.
    fn emit_center_ngon(&mut self, vertex: usize, j: usize, k: usize) -> Result<()> {
        let ring = self.bev(vertex)?.ring.clone();
        let repface = if self.bev(vertex)?.any_seam {
            self.frep_for_center_poly(vertex)?
        } else {
            None
        };
        let neighbors = repface.and_then(|face| incident_neighbors(self.polygons, face, vertex));
        let unsnapped = match repface {
            Some(face) => self.face_internal_boundverts(vertex, face)?,
            None => Vec::new(),
        };
        let mut vertices = Vec::with_capacity(ring.len());
        let mut sources = Vec::with_capacity(ring.len());
        let mut snaps = Vec::with_capacity(ring.len());
        for (i, bound) in ring.iter().enumerate() {
            let Some(id) = self.id(vertex, i, j, k)? else {
                return Ok(());
            };
            let source = repface.or_else(|| boundvert_rep_face(self.front, *bound));
            let snap = match (neighbors, repface) {
                (Some((first, second)), Some(_)) if !unsnapped.contains(bound) => {
                    let co = self
                        .grid_point(vertex, i, j, k)?
                        .unwrap_or(self.points[vertex]);
                    Some(closer_edge(self.points, co, vertex, first, second))
                }
                _ => None,
            };
            vertices.push(id);
            sources.push(source);
            snaps.push(snap);
        }
        if vertices.len() <= 2 {
            return Ok(());
        }
        let face = match repface.or(sources.first().copied().flatten()) {
            Some(face) => face,
            None => self.fallback_face(vertex)?,
        };
        self.emit_polygon(vertices, snaps, sources, face, Kind::Corner);
        Ok(())
    }

    /// The corners of `vertex`'s single ngon (Blender's `bevel_build_poly`): the boundary point and
    /// each profile point of every boundary vertex, with the seam-aware center representative.
    fn poly_corners(&mut self, vertex: usize) -> Result<(Vec<Corner>, Option<usize>)> {
        let ring = self.bev(vertex)?.ring.clone();
        let repface = if self.bev(vertex)?.any_seam {
            self.frep_for_center_poly(vertex)?
        } else {
            None
        };
        let neighbors = repface.and_then(|face| incident_neighbors(self.polygons, face, vertex));
        let unsnapped = match repface {
            Some(face) => self.face_internal_boundverts(vertex, face)?,
            None => Vec::new(),
        };
        let mut corners = Vec::new();
        for bound in ring {
            let i = self.position(bound);
            let source = repface.or(boundvert_rep_face(self.front, bound));
            let co = self.front.bounds[bound].co;
            let snap = match neighbors {
                Some((first, second)) if !unsnapped.contains(&bound) => {
                    Some(closer_edge(self.points, co, vertex, first, second))
                }
                _ => None,
            };
            if let Some(id) = self.id(vertex, i, 0, 0)? {
                corners.push((id, snap, source));
            }
            let Some(ebev) = self.front.bounds[bound].ebev else {
                continue;
            };
            let edge_seg = self.front.edges[ebev].seg;
            for k in 1..edge_seg {
                let Some(id) = self.id(vertex, i, 0, k)? else {
                    continue;
                };
                let snap = match neighbors {
                    Some((first, second)) if k >= edge_seg / 2 => {
                        let co = self.grid_point(vertex, i, 0, k)?.unwrap_or(co);
                        Some(closer_edge(self.points, co, vertex, first, second))
                    }
                    _ => None,
                };
                corners.push((id, snap, source));
            }
        }
        Ok((corners, repface))
    }

    /// Blender's `bevel_build_poly`, and for TRIFAN its `bevel_build_trifan` split of the ngon from
    /// its last corner into the fan `(last, k, k + 1)`.
    fn emit_vertex_ngon(&mut self, vertex: usize, fan: bool) -> Result<()> {
        let (corners, repface) = self.poly_corners(vertex)?;
        let count = corners.len();
        if count <= 2 {
            return Ok(());
        }
        let fallback = self.fallback_face(vertex)?;
        let face = repface.or(corners[0].2).unwrap_or(fallback);
        if !fan {
            let vertices = corners.iter().map(|c| c.0).collect();
            let snaps = corners.iter().map(|c| c.1).collect();
            let sources = corners.iter().map(|c| c.2).collect();
            self.emit_polygon(vertices, snaps, sources, face, Kind::Corner);
            return Ok(());
        }
        let last = corners[count - 1];
        for k in 0..count - 2 {
            let triangle = [corners[k], corners[k + 1], last];
            let vertices = triangle.iter().map(|c| c.0).collect();
            let snaps = triangle.iter().map(|c| c.1).collect();
            let sources = triangle.iter().map(|c| c.2).collect();
            self.emit_polygon(vertices, snaps, sources, face, Kind::Corner);
            let uv_mask = if count == 3 {
                0b111
            } else if k == 0 {
                0b101
            } else {
                0
            };
            self.emission.set_last_uv_registration(uv_mask)?;
        }
        Ok(())
    }

    /// Blender's `bevel_build_cutoff`: the corner points below each boundary vertex, a face closing
    /// each profile, and a bottom face through the corner points when they are distinct.
    fn emit_cutoff(&mut self, vertex: usize) -> Result<()> {
        let ring = self.bev(vertex)?.ring.clone();
        let count = ring.len();
        let seg = self.front.p.seg;
        let sqrt2 = 2.0_f32.sqrt();
        let normal = self.front.vertex_normals[vertex].as_vec3();
        let mut corner_co = Vec::with_capacity(count);
        for (i, bound) in ring.iter().enumerate() {
            let prev = ring[(i + count - 1) % count];
            let mut down = self.front.bounds[*bound]
                .profile
                .plane_no
                .as_vec3()
                .cross(self.front.bounds[prev].profile.plane_no.as_vec3());
            if down.dot(normal) > 0.0 {
                down = -down;
            }
            let sum = self.front.bounds[*bound].profile.height as f32 / sqrt2
                + self.front.bounds[prev].profile.height as f32 / sqrt2;
            let length = 0.5_f32 * sum;
            corner_co.push((self.front.bounds[*bound].co.as_vec3() + down * length).as_dvec3());
        }
        {
            let grid = self.grid_mut(vertex)?;
            for (i, co) in corner_co.iter().enumerate() {
                let index = grid.vm.index(i, 1, 0);
                grid.vm.co[index] = *co;
                grid.defined[index] = true;
                let prev_index = grid.vm.index((i + count - 1) % count, 1, 1);
                grid.vm.co[prev_index] = *co;
                grid.defined[prev_index] = true;
            }
        }
        let build_center = if count == 3 {
            let far = |a: usize, b: usize| {
                (corner_co[a].as_vec3() - corner_co[b].as_vec3()).length_squared() > EPSILON as f32
            };
            far(0, 1) && far(0, 2) && far(1, 2)
        } else {
            true
        };
        let corner_ids: Vec<usize> = if build_center {
            let mut ids = Vec::with_capacity(count);
            for i in 0..count {
                ids.push(
                    self.id(vertex, i, 1, 0)?
                        .ok_or_else(|| internal_invariant("corner point defined"))?,
                );
            }
            // Blender: `mesh_vert(prev, 1, 1)->v = mesh_vert(i, 1, 0)->v`, the second corner of each
            // profile being the next profile's first corner.
            for (i, id) in ids.iter().enumerate() {
                self.share_id(vertex, (i + count - 1) % count, 1, 1, *id)?;
            }
            ids
        } else {
            let shared = self
                .id(vertex, 0, 1, 0)?
                .ok_or_else(|| internal_invariant("corner point defined"))?;
            vec![shared; count]
        };
        let material_face = self.fallback_face(vertex)?;
        for (i, corner_id) in corner_ids.iter().enumerate().take(count) {
            let mut vertices = vec![*corner_id];
            for k in 0..=seg {
                vertices.push(
                    self.id(vertex, i, 0, k)?
                        .ok_or_else(|| internal_invariant("profile point defined"))?,
                );
            }
            if build_center {
                vertices.push(
                    self.id(vertex, i, 1, 1)?
                        .ok_or_else(|| internal_invariant("corner point defined"))?,
                );
            }
            let n = vertices.len();
            self.emit_polygon(
                vertices,
                vec![None; n],
                vec![None; n],
                material_face,
                Kind::Corner,
            );
        }
        if build_center {
            self.emit_polygon(
                corner_ids.clone(),
                vec![None; count],
                vec![None; count],
                material_face,
                Kind::Corner,
            );
        }
        Ok(())
    }

    /// Blender's `mesh_vert(...)->v = ...`: the cell `(i, j, k)` of `vertex` takes the output point `id`.
    fn share_id(&mut self, vertex: usize, i: usize, j: usize, k: usize, id: usize) -> Result<()> {
        if let Some(index) = self.grid(vertex)?.slot(i, j, k) {
            self.grid_mut(vertex)?.ids[index] = Some(id);
        }
        Ok(())
    }

    /// Blender's `bevel_extend_edge_data_ex`: extend each marked input-edge run over the outer
    /// profile boundary of this vertex mesh.
    fn extend_edge_data(&mut self, vertex: usize) -> Result<()> {
        if self.mesh_kinds[vertex] == Some(MeshKind::TriFan) || self.bev(vertex)?.selcount < 2 {
            return Ok(());
        }
        let ring = self.bev(vertex)?.ring.clone();
        for (start, bound) in ring.iter().enumerate() {
            let (seam_len, sharp_len) = (
                self.front.bounds[*bound].seam_len,
                self.front.bounds[*bound].sharp_len,
            );
            self.extend_edge_data_run(vertex, start, seam_len, false, true)?;
            self.extend_edge_data_run(vertex, start, sharp_len, true, false)?;
        }
        Ok(())
    }

    fn extend_edge_data_run(
        &mut self,
        vertex: usize,
        start: usize,
        extend_len: usize,
        sharp: bool,
        seam: bool,
    ) -> Result<()> {
        let count = self.bev(vertex)?.ring.len();
        if count == 0 {
            return Ok(());
        }
        let seg = self.front.p.seg;
        for step in 0..extend_len {
            let i = (start + step) % count;
            let next_i = (i + 1) % count;
            let Some(mut previous) = self.id(vertex, i, 0, 0)? else {
                continue;
            };
            let mut complete = true;
            for k in 1..seg {
                let Some(current) = self.id(vertex, i, 0, k)? else {
                    complete = false;
                    break;
                };
                if previous != current {
                    self.emission
                        .record_edge_flags(previous, current, sharp, seam);
                }
                previous = current;
            }
            if !complete {
                continue;
            }
            if let Some(next) = self.id(vertex, next_i, 0, 0)?
                && previous != next
            {
                self.emission.record_edge_flags(previous, next, sharp, seam);
            }
        }
        Ok(())
    }

    /// Blender's `bevel_build_edge_polygons`: the strip of quads along the beveled edge `(a, b)`.
    fn emit_strip(&mut self, a: usize, b: usize) -> Result<()> {
        let key = edge_key(a, b);
        if !self.front.geo.selected_edges.contains(&key)
            || self.front.verts[a].is_none()
            || self.front.verts[b].is_none()
        {
            return Ok(());
        }
        let (Some(e1_id), Some(e2_id)) = (self.edge_half(a, key), self.edge_half(b, key)) else {
            return Ok(());
        };
        let e1 = self.front.edges[e1_id].clone();
        let e2 = self.front.edges[e2_id].clone();
        if !e1.is_bev || !e2.is_bev {
            return Ok(());
        }
        let nseg = e1.seg;
        let (Some(left1), Some(right2), Some(left2)) = (e1.leftv, e2.rightv, e2.leftv) else {
            return Ok(());
        };
        let (f1, f2) = (e1.fprev, e1.fnext);
        let faces = [f1, f1, f2, f2];
        let i1 = self.position(left1);
        let i2 = self.position(left2);
        let mut verts = [
            self.id(a, i1, 0, 0)?,
            self.id(b, self.position(right2), 0, 0)?,
        ];
        let odd = nseg % 2 == 1;
        let mid = nseg / 2;
        let seam = e1.is_seam;
        let f_choice = if odd && seam {
            choose_rep_face(&[f1, f2], self.context, self.polygons, self.points)
        } else {
            None
        };
        let center_adj_k = if odd && seam && nseg > 1 {
            if f_choice == f1 { mid + 2 } else { mid }
        } else {
            usize::MAX
        };
        let rung_first = (verts[0], verts[1]);
        let edge = Some(self.edge_points(e1_id));
        for k in 1..=nseg {
            let top = [self.id(a, i1, 0, k)?, self.id(b, i2, 0, nseg - k)?];
            let (Some(v0), Some(v1), Some(v2), Some(v3)) = (verts[0], verts[1], top[1], top[0])
            else {
                return Ok(());
            };
            let (sources, snaps, face) = if odd && k == mid + 1 {
                if seam {
                    let snaps = if f_choice == f1 {
                        [None, None, edge, edge]
                    } else {
                        [edge, edge, None, None]
                    };
                    ([f_choice; 4], snaps, f_choice)
                } else {
                    (faces, [None; 4], f_choice.or(f1))
                }
            } else if odd && k == center_adj_k && seam {
                if k == mid {
                    ([f1; 4], [None, None, edge, edge], f1)
                } else {
                    ([f2; 4], [edge, edge, None, None], f2)
                }
            } else if !odd && k == mid {
                ([f1; 4], [None, None, edge, edge], f1)
            } else if !odd && k == mid + 1 {
                ([f2; 4], [edge, edge, None, None], f2)
            } else {
                let f = if k <= mid { f1 } else { f2 };
                ([f; 4], [None; 4], f)
            };
            let face = match face {
                Some(face) => face,
                None => self.fallback_face(a)?,
            };
            self.emit_polygon(
                vec![v0, v1, v2, v3],
                snaps.to_vec(),
                sources.to_vec(),
                face,
                Kind::Strip,
            );
            if k < nseg
                && let (Some(first), Some(second)) = (top[0], top[1])
            {
                self.emission.suppress_edge_flags(first, second);
            }
            verts = [top[0], top[1]];
        }
        // Blender's `bevel_build_edge_polygons` copies the beveled edge's attributes onto the first and
        // last rungs of the strip (`BM_elem_attrs_copy(bm, bme, bme1/bme2)`, bmesh_bevel.cc:7611-7612).
        // Rebuilt faces copy later and overwrite these.
        let (sharp, seam) = self.front.input_edge_flags_by_id(e1.edge_id);
        for (first, second) in [rung_first, (verts[0], verts[1])] {
            if let (Some(first), Some(second)) = (first, second) {
                self.emission
                    .set_edge_state(first, second, sharp, seam, Some(e1.edge_id));
            }
        }
        Ok(())
    }
    /// Blender's `bevel_reattach_wires`: connect each wire to the nearest new boundary points.
    fn reattach_wires(&mut self) -> Result<()> {
        let mut created = HashSet::new();
        for vertex in 0..self.front.verts.len() {
            if self.front.verts[vertex].is_none() {
                continue;
            }
            for index in 0..self.wire_edges[vertex].len() {
                let (a, b) = self.wire_edges[vertex][index];
                let other = if a == vertex { b } else { a };
                if self.front.verts[other].is_some() && other < vertex {
                    continue;
                }
                let (first, second) = if self.front.verts[other].is_some() {
                    let mut closest = (f32::MAX, None);
                    for first_bound in &self.bev(vertex)?.ring {
                        for second_bound in &self.bev(other)?.ring {
                            let distance = self.front.bounds[*first_bound]
                                .co
                                .as_vec3()
                                .distance_squared(self.front.bounds[*second_bound].co.as_vec3());
                            if distance < closest.0 {
                                closest = (distance, Some((*first_bound, *second_bound)));
                            }
                        }
                    }
                    let Some((first_bound, second_bound)) = closest.1 else {
                        continue;
                    };
                    let first_index = self.position(first_bound);
                    let second_index = self.position(second_bound);
                    let (Some(first), Some(second)) = (
                        self.id(vertex, first_index, 0, 0)?,
                        self.id(other, second_index, 0, 0)?,
                    ) else {
                        continue;
                    };
                    (first, second)
                } else {
                    let other_co = self.points[other].as_vec3();
                    let mut closest = (f32::MAX, None);
                    for bound in &self.bev(vertex)?.ring {
                        let distance = self.front.bounds[*bound]
                            .co
                            .as_vec3()
                            .distance_squared(other_co);
                        if distance < closest.0 {
                            closest = (distance, Some(*bound));
                        }
                    }
                    let Some(bound) = closest.1 else {
                        continue;
                    };
                    let index = self.position(bound);
                    let Some(first) = self.id(vertex, index, 0, 0)? else {
                        continue;
                    };
                    (first, other)
                };
                let output_key = edge_key(first, second);
                if first == second || !created.insert(output_key) {
                    continue;
                }
                let (sharp, seam) = self.front.input_edge_flags_between(a, b);
                if sharp || seam {
                    self.emission.record_edge_flags(first, second, sharp, seam);
                }
                self.loose.push([first, second]);
            }
        }
        Ok(())
    }

    /// Blender's `bev_rebuild_polygon`: `face` with each beveled corner replaced by the boundary and
    /// profile points around its vertex, interpolated from the face itself.
    fn emit_rebuilt_face(&mut self, face: usize) -> Result<Vec<EdgeState>> {
        let vertices = self.polygons[face].0.clone();
        let count = vertices.len();
        let mut out: Vec<usize> = Vec::new();
        // Blender's `ee`: the face corner whose outgoing edge each output point was built from.
        let mut sources: Vec<usize> = Vec::new();
        for c in 0..count {
            let v = vertices[c];
            if self.front.verts[v].is_none() {
                out.push(v);
                sources.push(c);
                continue;
            }
            let next = vertices[(c + 1) % count];
            let prev = vertices[(c + count - 1) % count];
            let (Some(e_id), Some(eprev_id)) = (
                self.edge_half(v, edge_key(v, next)),
                self.edge_half(v, edge_key(prev, v)),
            ) else {
                continue;
            };
            let points = self.corner_points(v, face, e_id, eprev_id)?;
            sources.extend(std::iter::repeat_n(c, points.len()));
            out.extend(points);
        }
        let states = self.copy_rebuilt_edge_flags(&out, &sources, &vertices);
        let n = out.len();
        self.emit_polygon(out, vec![None; n], vec![Some(face); n], face, Kind::Source);
        Ok(states)
    }

    /// Blender's attribute copy at the end of `bev_rebuild_polygon`: every new edge `out[k]`→`out[k+1]`
    /// that is not the source edge of `sources[k]` takes that edge's `(sharp, seam)`. Within one
    /// corner's run the copy keeps a seam or sharp mark only where the previous source edge also has
    /// it (`bmesh_bevel.cc` 7258–7285).
    fn copy_rebuilt_edge_flags(
        &mut self,
        out: &[usize],
        sources: &[usize],
        vertices: &[usize],
    ) -> Vec<EdgeState> {
        let n = out.len();
        if n == 0 {
            return Vec::new();
        }
        let count = vertices.len();
        let edge_of = |corner: usize| (vertices[corner], vertices[(corner + 1) % count]);
        let mut states = Vec::new();
        let mut previous = sources[n - 1];
        for k in 0..n {
            let (first, second) = (out[k], out[(k + 1) % n]);
            let (source_first, source_second) = edge_of(sources[k]);
            if edge_key(first, second) == edge_key(source_first, source_second) {
                // The rebuilt loop keeps this input `BMEdge`, so Blender copies nothing onto it.
                self.emission.record_kept_edge(first, second);
                continue;
            }
            let (mut sharp, mut seam) = self
                .front
                .input_edge_flags_between(source_first, source_second);
            if k + 1 < n && sources[k] == sources[k + 1] {
                let (previous_a, previous_b) = edge_of(previous);
                let (previous_sharp, previous_seam) =
                    self.front.input_edge_flags_between(previous_a, previous_b);
                seam &= previous_seam;
                sharp &= previous_sharp;
            } else {
                previous = sources[k];
            }
            let source = self.front.input_edge_id(source_first, source_second);
            states.push(((first, second), sharp, seam, source));
        }
        states
    }

    /// Blender's loop walk of `bev_rebuild_polygon` for one beveled corner of `face` at `vertex`,
    /// from the edge half `eprev` to `e` around the vertex.
    fn corner_points(
        &mut self,
        vertex: usize,
        face: usize,
        e_id: usize,
        eprev_id: usize,
    ) -> Result<Vec<usize>> {
        let seg = self.front.p.seg;
        let e = self.front.edges[e_id].clone();
        let eprev = self.front.edges[eprev_id].clone();
        // The direction around `vertex` that matches the face's orientation.
        let go_ccw = if e.prev == eprev_id {
            if eprev.prev == e_id {
                e.fnext != Some(face)
            } else {
                true
            }
        } else if eprev.prev == e_id {
            false
        } else {
            self.count_ccw_between(eprev_id, e_id) < self.count_ccw_between(e_id, eprev_id)
        };
        let mut on_profile_start = false;
        let (start, end) = if go_ccw {
            let mut start = eprev.rightv;
            if e.profile_index > 0 {
                start = start.map(|id| self.front.bounds[id].prev);
                on_profile_start = true;
            }
            (start, e.leftv)
        } else {
            let mut start = eprev.leftv;
            if eprev.profile_index > 0 {
                start = start.map(|id| self.front.bounds[id].next);
                on_profile_start = true;
            }
            (start, e.rightv)
        };
        let (Some(mut current), Some(end)) = (start, end) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        if !on_profile_start && let Some(id) = self.id(vertex, self.position(current), 0, 0)? {
            out.push(id);
        }
        while current != end {
            if go_ccw {
                let i = self.position(current);
                let kstart = if on_profile_start { e.profile_index } else { 1 };
                on_profile_start = false;
                let kend = if eprev.rightv == Some(current) && eprev.profile_index > 0 {
                    eprev.profile_index
                } else {
                    seg
                };
                for k in kstart..=kend {
                    if let Some(id) = self.id(vertex, i, 0, k)? {
                        out.push(id);
                    }
                }
                current = self.front.bounds[current].next;
            } else {
                let previous = self.front.bounds[current].prev;
                let i = self.position(previous);
                let kstart = if on_profile_start {
                    eprev.profile_index
                } else {
                    seg - 1
                };
                on_profile_start = false;
                let kend = if e.rightv == Some(previous) && e.profile_index > 0 {
                    e.profile_index
                } else {
                    0
                };
                let mut k = kstart as isize;
                while k >= kend as isize {
                    if let Some(id) = self.id(vertex, i, 0, k as usize)? {
                        out.push(id);
                    }
                    k -= 1;
                }
                current = previous;
            }
        }
        Ok(out)
    }

    /// Blender's `count_ccw_edges_between`: how many steps around `vertex` from `from` to `to`.
    fn count_ccw_between(&self, from: usize, to: usize) -> usize {
        let mut count = 0;
        let mut e = from;
        loop {
            if e == to {
                break;
            }
            e = self.front.edges[e].next;
            count += 1;
            if e == from {
                break;
            }
        }
        count
    }
}

/// Blender's `BM_FACES_OF_VERT` walk over the input `BMesh` (`bmesh_iterators.cc`, `bmiter__face_of_vert_*`
/// with `bmesh_structure.cc`). A vertex's edges form a disk cycle in ascending edge index from its first
/// edge `v->e`; each edge's loops form a radial cycle, headed by the loop of its last-created face. `BMesh`
/// creates faces and edges in mesh order, so the input indices give that order. Faces are yielded per
/// loop at the vertex, as the iterator does.
struct FaceWalk<'a> {
    faces: &'a [Polygon],
    /// Per face and corner, the mesh edge index of the corner's outgoing edge.
    loop_edges: Vec<Vec<Option<usize>>>,
    /// Per mesh edge, its loops `(face, corner)` in ascending face order.
    radial: Vec<Vec<(usize, usize)>>,
    /// Per vertex, its incident mesh edges in ascending index order.
    disk: Vec<Vec<usize>>,
}

impl<'a> FaceWalk<'a> {
    fn new(front: &Front, faces: &'a [Polygon]) -> Self {
        let mut loop_edges = Vec::with_capacity(faces.len());
        let mut radial = vec![Vec::new(); front.geo.edge_order.len()];
        for (face, (vertices, _)) in faces.iter().enumerate() {
            let count = vertices.len();
            let mut row = Vec::with_capacity(count);
            for corner in 0..count {
                let edge = front.input_edge_id(vertices[corner], vertices[(corner + 1) % count]);
                if let Some(edge) = edge {
                    radial[edge].push((face, corner));
                }
                row.push(edge);
            }
            loop_edges.push(row);
        }
        let mut disk = vec![Vec::new(); front.verts.len()];
        for (edge, &(first, second)) in front.geo.edge_order.iter().enumerate() {
            disk[first].push(edge);
            if second != first {
                disk[second].push(edge);
            }
        }
        Self {
            faces,
            loop_edges,
            radial,
            disk,
        }
    }

    fn vertex(&self, (face, corner): (usize, usize)) -> usize {
        self.faces[face].0[corner]
    }

    /// `bmesh_radial_faceloop_find_next`: the next loop of the loop's edge at `vertex` in the radial cycle,
    /// or the loop itself when there is none.
    fn radial_next(&self, loop_: (usize, usize), vertex: usize) -> (usize, usize) {
        let Some(edge) = self.loop_edges[loop_.0][loop_.1] else {
            return loop_;
        };
        let cycle = &self.radial[edge];
        let Some(position) = cycle.iter().position(|candidate| *candidate == loop_) else {
            return loop_;
        };
        for step in 1..cycle.len() {
            let candidate = cycle[(position + step) % cycle.len()];
            if self.vertex(candidate) == vertex {
                return candidate;
            }
        }
        loop_
    }

    /// `bmesh_radial_faceloop_find_first` from the edge's radial head: its first loop at `vertex`.
    fn radial_first(&self, edge: usize, vertex: usize) -> Option<(usize, usize)> {
        let cycle = &self.radial[edge];
        let head = cycle.len().checked_sub(1)?;
        (0..cycle.len())
            .map(|step| cycle[(head + step) % cycle.len()])
            .find(|candidate| self.vertex(*candidate) == vertex)
    }

    fn disk_next(&self, edge: usize, vertex: usize) -> usize {
        let disk = &self.disk[vertex];
        let position = disk
            .iter()
            .position(|candidate| *candidate == edge)
            .unwrap_or(0);
        disk[(position + 1) % disk.len()]
    }

    /// `bmesh_disk_faceedge_find_next`: the next edge around `vertex` with a loop at it, or `edge` itself.
    fn disk_face_edge_next(&self, edge: usize, vertex: usize) -> usize {
        let mut current = self.disk_next(edge, vertex);
        while current != edge {
            if self.radial_first(current, vertex).is_some() {
                return current;
            }
            current = self.disk_next(current, vertex);
        }
        edge
    }

    /// `bmesh_disk_faceloop_find_first`: the first loop at `vertex` from the disk head `v->e`.
    fn first_loop(&self, vertex: usize) -> Option<(usize, usize)> {
        for &edge in &self.disk[vertex] {
            if let Some(&head) = self.radial[edge].last() {
                return Some(if self.vertex(head) == vertex {
                    head
                } else {
                    let count = self.faces[head.0].0.len();
                    (head.0, (head.1 + 1) % count)
                });
            }
        }
        None
    }

    /// The faces of `vertex` in the order `BM_FACES_OF_VERT` yields them.
    fn faces_of_vertex(&self, vertex: usize) -> Vec<usize> {
        let mut count: usize = self.disk[vertex]
            .iter()
            .map(|&edge| {
                self.radial[edge]
                    .iter()
                    .filter(|loop_| self.vertex(**loop_) == vertex)
                    .count()
            })
            .sum();
        let Some(first) = self.first_loop(vertex) else {
            return Vec::new();
        };
        let Some(mut edge_next) = self.loop_edges[first.0][first.1] else {
            return Vec::new();
        };
        let mut loop_first = first;
        let mut loop_next = Some(first);
        let mut faces = Vec::with_capacity(count);
        while let Some(current) = loop_next {
            faces.push(current.0);
            if count > 0 {
                count -= 1;
                let mut next = self.radial_next(current, vertex);
                if next == loop_first {
                    edge_next = self.disk_face_edge_next(edge_next, vertex);
                    let Some(first_here) = self.radial_first(edge_next, vertex) else {
                        break;
                    };
                    loop_first = first_here;
                    next = first_here;
                }
                loop_next = Some(next);
            }
            if count == 0 {
                loop_next = None;
            }
        }
        faces
    }
}

/// Builds the output of a bevel: vertex meshes, strips along beveled edges, and the rebuilt faces.
pub(super) fn bevel_output(
    front: &Front,
    points: &[DVec3],
    polygons: &[Polygon],
    context: &Context,
) -> Result<(Emission, Vec<[usize; 2]>)> {
    let mut builder = Builder::new(front, points, polygons, context);
    let beveled: Vec<usize> = (0..front.verts.len())
        .filter(|vertex| front.verts[*vertex].is_some())
        .collect();
    for vertex in &beveled {
        builder.build_grid(*vertex)?;
    }
    for vertex in &beveled {
        let vertex = *vertex;
        match builder.mesh_kinds[vertex] {
            Some(MeshKind::Poly) => builder.emit_vertex_ngon(vertex, false)?,
            Some(MeshKind::TriFan) => builder.emit_vertex_ngon(vertex, true)?,
            Some(MeshKind::Adj) => builder.emit_adj(vertex)?,
            Some(MeshKind::Cutoff) => builder.emit_cutoff(vertex)?,
            Some(MeshKind::None) => builder.emit_two_edge(vertex)?,
            None => {}
        }
    }
    if !front.p.affect_vertices {
        for (a, b) in &front.geo.edge_order {
            builder.emit_strip(*a, *b)?;
        }
    }
    for vertex in &beveled {
        builder.extend_edge_data(*vertex)?;
    }
    let mut rebuilt_states: Vec<Option<Vec<EdgeState>>> = vec![None; polygons.len()];
    for (face, (vertices, _)) in polygons.iter().enumerate() {
        if vertices.iter().any(|v| front.verts[*v].is_some()) {
            rebuilt_states[face] = Some(builder.emit_rebuilt_face(face)?);
        } else {
            let n = vertices.len();
            builder.emit_polygon(
                vertices.clone(),
                vec![None; n],
                vec![Some(face); n],
                face,
                Kind::Original,
            );
        }
    }
    // Blender rebuilds each face once, scanning the beveled vertices in index order and the faces
    // around each in `BM_FACES_OF_VERT` order. Its copies overwrite one another on shared rungs, so the
    // last copy applies.
    let walk = FaceWalk::new(front, polygons);
    for vertex in (0..front.verts.len()).filter(|vertex| front.verts[*vertex].is_some()) {
        for face in walk.faces_of_vertex(vertex) {
            if let Some(states) = rebuilt_states[face].take() {
                for ((first, second), sharp, seam, source) in states {
                    builder
                        .emission
                        .set_edge_state(first, second, sharp, seam, source);
                }
            }
        }
    }
    if !front.p.affect_vertices {
        builder.reattach_wires()?;
    }
    Ok((builder.emission, builder.loose))
}
