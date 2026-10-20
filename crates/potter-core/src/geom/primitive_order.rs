//! Blender's `BMesh` element order for the mesh primitives.
//!
//! References (Blender 5.2.2): `bmesh/operators/bmo_primitive.cc` (`bmo_create_grid_exec`,
//! `bmo_create_circle_exec`, `bmo_create_cone_exec`, `bmo_create_cube_exec`) and
//! `bmesh/intern/bmesh_construct.cc` (`BM_edges_from_verts_ensure`, reached through
//! `BM_face_create_verts` with `create_edges`).
//!
//! `BMesh` appends each vertex, edge and face in creation order, and a face creates its missing
//! boundary edges closing edge first. The operators also delete temporary elements after building
//! (cap centres, fan spokes, collapsed apex vertices, dissolved cap fans). The builders here replay
//! the creation sequence and then apply those deletions, so survivors keep their creation order.
//! Each rule was checked against `bpy.ops.mesh.primitive_*_add` output (`tests/fixtures/primitive_uvs`).

use std::collections::HashSet;

use glam::DVec3;

use super::{CircleFillType, Edge, EndFillType, Face, Mesh, MeshError, Vertex};

/// Elements in creation order, indexed into `vertices`.
#[derive(Default)]
pub(super) struct Elements {
    vertices: Vec<DVec3>,
    edges: Vec<[usize; 2]>,
    faces: Vec<Vec<usize>>,
    edge_keys: HashSet<(usize, usize)>,
}

impl Elements {
    /// `BM_vert_create`: appends a vertex and returns its index.
    fn vertex(&mut self, co: DVec3) -> usize {
        self.vertices.push(co);
        self.vertices.len() - 1
    }

    /// `BM_edge_create` with `BM_CREATE_NO_DOUBLE`: appends the edge unless one already joins the pair.
    fn edge(&mut self, first: usize, second: usize) {
        if self.edge_keys.insert(super::edge_key(first, second)) {
            self.edges.push([first, second]);
        }
    }

    /// `BM_face_create_verts` with `create_edges`: each missing boundary edge is created from the
    /// previous corner to the next, so the closing edge (last corner to first) comes first.
    fn face(&mut self, corners: &[usize]) {
        self.add_face(corners, false);
    }

    /// Like [`Self::face`], but each edge is stored with its lower vertex first, as the torus
    /// primitive's `from_pydata` mesh stores them.
    fn face_lower_first(&mut self, corners: &[usize]) {
        self.add_face(corners, true);
    }

    fn add_face(&mut self, corners: &[usize], lower_first: bool) {
        let Some(&last) = corners.last() else {
            return;
        };
        let mut previous = last;
        for &corner in corners {
            if lower_first {
                self.edge(previous.min(corner), previous.max(corner));
            } else {
                self.edge(previous, corner);
            }
            previous = corner;
        }
        self.faces.push(corners.to_vec());
    }

    /// Converts the surviving elements into a mesh, with IDs assigned in storage order.
    fn into_mesh(self) -> Result<Mesh, MeshError> {
        let id = |index: usize| u32::try_from(index).map_err(|_| MeshError::IdExhausted);
        let mut mesh = Mesh::new();
        mesh.vertices = self
            .vertices
            .into_iter()
            .enumerate()
            .map(|(index, co)| Ok(Vertex { id: id(index)?, co }))
            .collect::<Result<_, MeshError>>()?;
        mesh.edges = self
            .edges
            .into_iter()
            .enumerate()
            .map(|(index, [first, second])| {
                Ok(Edge {
                    id: id(index)?,
                    vertices: [id(first)?, id(second)?],
                })
            })
            .collect::<Result<_, MeshError>>()?;
        mesh.faces = self
            .faces
            .into_iter()
            .enumerate()
            .map(|(index, corners)| {
                Ok(Face {
                    id: id(index)?,
                    vertices: corners
                        .into_iter()
                        .map(id)
                        .collect::<Result<_, MeshError>>()?,
                    material_index: 0,
                })
            })
            .collect::<Result<_, MeshError>>()?;
        mesh.next_id.vertex = id(mesh.vertices.len())?;
        mesh.next_id.edge = id(mesh.edges.len())?;
        mesh.next_id.face = id(mesh.faces.len())?;
        Ok(mesh)
    }
}

/// `sin_cos_from_fraction(index, count)`: the sine and cosine of `2 pi index / count`.
#[expect(
    clippy::cast_precision_loss,
    reason = "indices derive from u32 counts and are exactly representable in f64"
)]
fn sin_cos_fraction(index: usize, count: usize) -> (f64, f64) {
    let angle = std::f64::consts::TAU * index as f64 / count as f64;
    (angle.sin(), angle.cos())
}

/// A grid of `nx` by `ny` vertices, row by row from `y = -size_y / 2`, `bmo_create_grid_exec`.
#[expect(
    clippy::cast_precision_loss,
    reason = "grid indices derive from u32 counts and are exactly representable in f64"
)]
pub(super) fn grid(nx: usize, ny: usize, size_x: f64, size_y: f64) -> Result<Mesh, MeshError> {
    let mut elements = Elements::default();
    let mut index = vec![0; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let px = -size_x * 0.5 + size_x * x as f64 / (nx - 1) as f64;
            let py = -size_y * 0.5 + size_y * y as f64 / (ny - 1) as f64;
            index[y * nx + x] = elements.vertex(DVec3::new(px, py, 0.0));
        }
    }
    for y in 0..ny - 1 {
        for x in 0..nx - 1 {
            let lower_left = y * nx + x;
            elements.face(&[
                index[lower_left],
                index[lower_left + 1],
                index[lower_left + nx + 1],
                index[lower_left + nx],
            ]);
        }
    }
    elements.into_mesh()
}

/// A circle in the XY plane starting at `(0, r)` and running counter-clockwise,
/// `bmo_create_circle_exec`. The ring edges are created as the ring is walked; a fan adds its
/// triangles in the same loop, and an N-gon fill is the ring itself.
pub(super) fn circle(
    segments: usize,
    radius: f64,
    fill_type: CircleFillType,
) -> Result<Mesh, MeshError> {
    let mut elements = Elements::default();
    let fan = fill_type == CircleFillType::TriFan;
    let center = fan.then(|| elements.vertex(DVec3::ZERO));
    let mut ring = Vec::with_capacity(segments);
    for index in 0..segments {
        let (sin, cos) = sin_cos_fraction(index, segments);
        let vertex = elements.vertex(DVec3::new(-radius * sin, radius * cos, 0.0));
        if index > 0 {
            elements.edge(vertex, ring[index - 1]);
            if let Some(center) = center {
                elements.face(&[center, ring[index - 1], vertex]);
            }
        }
        ring.push(vertex);
    }
    elements.edge(ring[0], ring[segments - 1]);
    match fill_type {
        CircleFillType::Nothing => {}
        CircleFillType::Ngon => elements.faces.push(ring),
        CircleFillType::TriFan => {
            if let Some(center) = center {
                elements.face(&[center, ring[segments - 1], ring[0]]);
            }
        }
    }
    elements.into_mesh()
}

/// Which end a cone collapses to a single apex vertex, or neither.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Collapse {
    None,
    /// `radius2 == 0`: the top ring collapses to its first vertex.
    Top,
    /// `radius1 == 0`: the bottom ring collapses to its first vertex.
    Bottom,
}

/// The rings and centres of a cylinder or cone after `bmo_create_cone_exec` creates them.
struct ConeRings {
    elements: Elements,
    centres: Option<(usize, usize)>,
    bottoms: Vec<usize>,
    tops: Vec<usize>,
}

/// Replays `bmo_create_cone_exec`'s creation sequence: both cap centres, then each segment's bottom
/// and top vertex, the cap fans and side quad of each segment, and the closing fans and side.
fn cone_rings(segments: usize, radius1: f64, radius2: f64, half: f64, cap: bool) -> ConeRings {
    let mut elements = Elements::default();
    let centres = cap.then(|| {
        let bottom = elements.vertex(DVec3::new(0.0, 0.0, -half));
        let top = elements.vertex(DVec3::new(0.0, 0.0, half));
        (bottom, top)
    });
    let mut bottoms = Vec::with_capacity(segments);
    let mut tops = Vec::with_capacity(segments);
    for index in 0..segments {
        let (sin, cos) = sin_cos_fraction(index, segments);
        let v1 = elements.vertex(DVec3::new(radius1 * sin, radius1 * cos, -half));
        let v2 = elements.vertex(DVec3::new(radius2 * sin, radius2 * cos, half));
        if index > 0 {
            if let Some((c1, c2)) = centres {
                elements.face(&[c1, bottoms[index - 1], v1]);
                elements.face(&[c2, v2, tops[index - 1]]);
            }
            elements.face(&[bottoms[index - 1], tops[index - 1], v2, v1]);
        }
        bottoms.push(v1);
        tops.push(v2);
    }
    if let Some((c1, c2)) = centres {
        elements.face(&[c1, bottoms[segments - 1], bottoms[0]]);
        elements.face(&[c2, tops[0], tops[segments - 1]]);
    }
    elements.face(&[
        bottoms[segments - 1],
        tops[segments - 1],
        tops[0],
        bottoms[0],
    ]);
    ConeRings {
        elements,
        centres,
        bottoms,
        tops,
    }
}

/// A cylinder or cone (`bmo_create_cone_exec`). A zero radius collapses its end to one apex, and
/// the operator then deletes the killed centre and dissolves the N-gon caps' fans.
pub(super) fn cone(
    segments: usize,
    radius1: f64,
    radius2: f64,
    depth: f64,
    fill: EndFillType,
) -> Result<Mesh, MeshError> {
    let collapse = match (radius1 == 0.0, radius2 == 0.0) {
        (false, true) => Collapse::Top,
        (true, false) => Collapse::Bottom,
        _ => Collapse::None,
    };
    let cap = fill != EndFillType::Nothing;
    let ngon = fill == EndFillType::Ngon;
    let rings = cone_rings(segments, radius1, radius2, depth * 0.5, cap);
    if collapse == Collapse::None && !ngon {
        return rings.elements.into_mesh();
    }

    // Centres Blender deletes: a collapsed end's killed centre, and both centres of an N-gon fill
    // (their fans are dissolved). Spokes and fan triangles touching them disappear.
    let mut removed = Vec::new();
    if let Some((c1, c2)) = rings.centres {
        if ngon {
            removed.extend([c1, c2]);
        } else if collapse == Collapse::Top {
            removed.push(c2);
        } else if collapse == Collapse::Bottom {
            removed.push(c1);
        }
    }
    assemble_cone(&rings, segments, collapse, ngon, &removed)
}

/// Survivors, edges and faces of a collapsed or dissolved cone, then its N-gon caps.
fn assemble_cone(
    rings: &ConeRings,
    segments: usize,
    collapse: Collapse,
    ngon: bool,
    removed: &[usize],
) -> Result<Mesh, MeshError> {
    let collapsed_ring = match collapse {
        Collapse::Top => &rings.tops,
        Collapse::Bottom | Collapse::None => &rings.bottoms,
    };
    let apex = collapsed_ring[0];
    let is_collapsed =
        |vertex: usize| collapse != Collapse::None && collapsed_ring.contains(&vertex);
    let target = |vertex: usize| if is_collapsed(vertex) { apex } else { vertex };

    // Survivors keep creation order. A collapsed top's apex moves to the end; a collapsed bottom's
    // apex keeps its place.
    let mut survivors: Vec<usize> = (0..rings.elements.vertices.len())
        .filter(|vertex| !removed.contains(vertex))
        .filter(|&vertex| !is_collapsed(vertex) || (vertex == apex && collapse == Collapse::Bottom))
        .collect();
    if collapse == Collapse::Top {
        survivors.push(apex);
    }
    let mut remap = vec![usize::MAX; rings.elements.vertices.len()];
    for (new, &old) in survivors.iter().enumerate() {
        remap[old] = new;
    }

    let mut result = Elements {
        vertices: survivors
            .iter()
            .map(|&old| rings.elements.vertices[old])
            .collect(),
        ..Elements::default()
    };
    let mut seen = HashSet::new();
    for &[first, second] in &rings.elements.edges {
        if removed.contains(&first) || removed.contains(&second) {
            continue;
        }
        let (a, b) = (target(first), target(second));
        if a == b || !seen.insert(super::edge_key(a, b)) {
            continue;
        }
        result.edges.push([remap[a], remap[b]]);
    }
    let mut faces: Vec<Vec<usize>> = Vec::with_capacity(rings.elements.faces.len() + 1);
    for face in &rings.elements.faces {
        if face.iter().any(|vertex| removed.contains(vertex)) {
            continue;
        }
        let mut corners: Vec<usize> = Vec::with_capacity(face.len());
        for &vertex in face {
            let vertex = target(vertex);
            if corners.last() != Some(&vertex) {
                corners.push(vertex);
            }
        }
        while corners.len() > 1 && corners.first() == corners.last() {
            corners.pop();
        }
        if corners.len() >= 3 {
            faces.push(corners.iter().map(|&vertex| remap[vertex]).collect());
        }
    }

    if ngon {
        // The dissolved caps are `BM_faces_join` results. A bottom cap walks its ring upward from
        // the first vertex. A top cap starts at the second vertex, then the first, then descends.
        // A cap that survives alone sits at the second-to-last side position; with two caps the top
        // takes that position and the bottom comes last.
        let bottom_cap = || {
            rings
                .bottoms
                .iter()
                .map(|&vertex| remap[vertex])
                .collect::<Vec<_>>()
        };
        let top_cap = || {
            std::iter::once(rings.tops[1])
                .chain(std::iter::once(rings.tops[0]))
                .chain((2..segments).rev().map(|index| rings.tops[index]))
                .map(|vertex| remap[vertex])
                .collect::<Vec<_>>()
        };
        let position = segments - 2;
        match collapse {
            Collapse::Top => faces.insert(position, bottom_cap()),
            Collapse::Bottom => faces.insert(position, top_cap()),
            Collapse::None => {
                faces.insert(position, top_cap());
                faces.push(bottom_cap());
            }
        }
    }
    result.faces = faces;
    result.into_mesh()
}

/// Polygons built face by face in `BMesh` order (`BM_face_create_verts` with `create_edges`), for
/// primitives whose edges follow their faces: the closing edge of each face is created first.
pub(super) fn polygons(positions: Vec<DVec3>, polygons: &[Vec<usize>]) -> Result<Mesh, MeshError> {
    let mut elements = Elements::default();
    for position in positions {
        elements.vertex(position);
    }
    for polygon in polygons {
        elements.face(polygon);
    }
    elements.into_mesh()
}

/// Polygons whose edges are given explicitly in Blender's creation order (`subdivide_edges`). Every
/// boundary edge of a polygon must appear in `edges`; a missing one is an error, not a silent insert.
pub(super) fn explicit_edges(
    positions: Vec<DVec3>,
    edges: &[[usize; 2]],
    polygons: &[Vec<usize>],
) -> Result<Mesh, MeshError> {
    let mut elements = Elements::default();
    for position in positions {
        elements.vertex(position);
    }
    for &[first, second] in edges {
        elements.edge(first, second);
    }
    for polygon in polygons {
        for (index, &corner) in polygon.iter().enumerate() {
            let next = polygon[(index + 1) % polygon.len()];
            if !elements.edge_keys.contains(&super::edge_key(corner, next)) {
                return Err(MeshError::InvalidTopology(
                    "face boundary edge is missing from the primitive's edge order",
                ));
            }
        }
        elements.faces.push(polygon.clone());
    }
    elements.into_mesh()
}

/// A torus (`add_torus` in `scripts/startup/bl_operators/add_mesh_torus.py`): its vertices and faces
/// are given, and its edges follow the closing-first face walk, each stored lower vertex first.
pub(super) fn torus(positions: Vec<DVec3>, polygons: &[Vec<usize>]) -> Result<Mesh, MeshError> {
    let mut elements = Elements::default();
    for position in positions {
        elements.vertex(position);
    }
    for polygon in polygons {
        elements.face_lower_first(polygon);
    }
    elements.into_mesh()
}

/// The unit box of `bmo_create_cube_exec` with corner `(x, y, z)` signs enumerated x, then y, then z.
pub(super) fn cube(half: DVec3) -> Result<Mesh, MeshError> {
    const CORNER_SIGNS: [[f64; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ];
    const FACES: [[usize; 4]; 6] = [
        [0, 1, 3, 2],
        [2, 3, 7, 6],
        [6, 7, 5, 4],
        [4, 5, 1, 0],
        [2, 6, 4, 0],
        [7, 3, 1, 5],
    ];
    let mut elements = Elements::default();
    let corners: Vec<usize> = CORNER_SIGNS
        .iter()
        .map(|sign| {
            elements.vertex(DVec3::new(
                sign[0] * half.x,
                sign[1] * half.y,
                sign[2] * half.z,
            ))
        })
        .collect();
    for face in FACES {
        elements.face(&face.map(|corner| corners[corner]));
    }
    elements.into_mesh()
}
